#!/usr/bin/env python3
"""Reasoning/tool session checks with scripted model tokens and real C tools.

With --model, compare all rendered prompts against that GGUF's actual Jinja
chat template. Otherwise use the tiny model and an independent Python renderer.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from gguf import GGUFReader
from jinja2 import Environment, BaseLoader
import futhark_server
import parity


def html_json(value):
    return (json.dumps(value, sort_keys=True).replace('<', '\\u003c').replace('>', '\\u003e')
            .replace('&', '\\u0026').replace("'", '\\u0027'))


def reference_render(messages, generation, thinking, tools, template=None):
    if template is not None:
        return template.render(messages=messages, tools=tools, add_generation_prompt=generation,
                               enable_thinking=thinking)
    # Independent reference for tiny GGUF tests; full-model mode uses Jinja.
    first_system = messages and messages[0]['role'] == 'system'
    if tools:
        out = '<|im_start|>system\n'
        if first_system:
            out += messages[0]['content'] + '\n\n'
        out += ('# Tools\n\nYou may call one or more functions to assist with the user query.\n\n'
                'You are provided with function signatures within <tools></tools> XML tags:\n<tools>')
        out += ''.join('\n' + html_json(tool) for tool in tools)
        out += ('\n</tools>\n\nFor each function call, return a json object with function name and arguments '
                'within <tool_call></tool_call> XML tags:\n<tool_call>\n'
                '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n')
    else:
        out = '<|im_start|>system\n' + messages[0]['content'] + '<|im_end|>\n' if first_system else ''
    last_user = len(messages) - 1
    for i, m in enumerate(messages):
        if m['role'] == 'user' and not (m['content'].startswith('<tool_response>') and m['content'].endswith('</tool_response>')):
            last_user = i
    for i, m in enumerate(messages):
        role, content = m['role'], m['content']
        if role == 'user' or (role == 'system' and i > 0):
            out += '<|im_start|>' + role + '\n' + content + '<|im_end|>\n'
        elif role == 'assistant':
            reason = m.get('reasoning_content')
            if reason is None:
                if '</think>' in content:
                    reason = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
                    content = content.split('</think>')[-1].lstrip('\n')
                else:
                    reason = ''
            out += '<|im_start|>assistant\n'
            if i > last_user and (i == len(messages) - 1 or reason):
                out += '<think>\n' + reason.strip('\n') + '\n</think>\n\n' + content.lstrip('\n')
            else:
                out += content
            for j, call in enumerate(m.get('tool_calls', [])):
                call = call.get('function', call)
                args = call['arguments'] if isinstance(call['arguments'], str) else html_json(call['arguments'])
                out += ('\n' if j > 0 or content else '') + '<tool_call>\n{"name": "' + call['name'] + '", "arguments": ' + args + '}\n</tool_call>'
            out += '<|im_end|>\n'
        elif role == 'tool':
            if i == 0 or messages[i - 1]['role'] != 'tool':
                out += '<|im_start|>user'
            out += '\n<tool_response>\n' + content + '\n</tool_response>'
            if i == len(messages) - 1 or messages[i + 1]['role'] != 'tool':
                out += '<|im_end|>\n'
    if generation:
        out += '<|im_start|>assistant\n' + ('' if thinking else '<think>\n\n</think>\n\n')
    return out


def message(role, content, reason=None, calls=None):
    return dict(role=role, content=content, reasoning_content=reason, tool_calls=calls or [])


def parse_output(tokenizer, ids, complete):
    # Independent token-state oracle, following the Python host's captures.
    tags = {tokenizer.token_to_id(name): name for name in ['<think>', '</think>', '<tool_call>', '</tool_call>']}
    mode, answer, pending, reason, calls, warnings = 'answer', [], [], None, [], []
    invalid = False
    for token in ids:
        tag = tags.get(token)
        if mode == 'reason':
            if tag == '</think>':
                reason = tokenizer.decode(pending, skip_special_tokens=False).strip()
                pending, mode = [], 'answer'
            else:
                pending.append(token)
        elif mode == 'tool':
            if tag == '</tool_call>':
                if not invalid:
                    try:
                        def unique(pairs):
                            obj = {}
                            for key, value in pairs:
                                if key in obj:
                                    raise ValueError('duplicate key')
                                obj[key] = value
                            return obj
                        call = json.loads(tokenizer.decode(pending, skip_special_tokens=False).strip(), object_pairs_hook=unique)
                        call = call.get('function', call)
                        assert isinstance(call['name'], str) and call['name']
                        assert isinstance(call['arguments'], (dict, str))
                        calls.append(call)
                    except (ValueError, KeyError, AssertionError, AttributeError, TypeError):
                        warnings.append('invalid')
                pending, mode, invalid = [], 'answer', False
            else:
                if tag == '<tool_call>':
                    invalid = True
                    warnings.append('nested')
                pending.append(token)
        elif tag == '<think>':
            mode, pending = 'reason', []
        elif tag == '<tool_call>':
            mode, pending = 'tool', []
        elif tag in ['</think>', '</tool_call>']:
            warnings.append('unmatched')
        else:
            answer.append(token)
    if mode == 'reason':
        reason = tokenizer.decode(pending, skip_special_tokens=False).strip()
        warnings.append('unclosed')
    if mode == 'tool':
        warnings.append('unclosed')
    if not complete and calls:
        warnings.append('incomplete')
    accepted = calls if complete and not warnings else []
    return message('assistant', tokenizer.decode(answer, skip_special_tokens=False).strip(), reason, accepted), answer, bool(warnings)


def tool_result(server, enabled, call):
    def error(text):
        return html_json({'error': text})
    if not enabled:
        return error('tools are disabled')
    names = {'sumN1': 'tool_sumN1_sum_of_first_n_integers', 'sumN2': 'tool_sumN2_sum_of_squares_of_first_n_integers'}
    if call['name'] not in names:
        return error('unknown tool: ' + call['name'])
    args = call['arguments']
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None  # Only diagnostic text varies for invalid encoded JSON.
    n = args.get('n') if isinstance(args, dict) else None
    if type(n) is not int:
        return error('tool argument n must be an integer')
    if not -(2**63) <= n < 2**63:
        return error('tool argument n is outside the i64 range')
    if not 0 <= n <= 1000000:
        return error('tool argument n must be between 0 and 1000000')
    server.put_value('n', np.int64(n))
    server.cmd_call(names[call['name']], 'result', 'n')
    value = int(server.get_value('result'))
    server.cmd_free('result')
    server.cmd_free('n')
    return str(value)


def expected_session(test, tokenizer, eos, tools, server, template):
    history = copy.deepcopy(test.get('messages', []))
    cached, prompts, executions, visible, saw_warning = [], [], [], '', False
    for turn in test['turns']:
        enabled, thinking, show = (turn.get(k, False) for k in ['tools', 'thinking', 'show'])
        history.append(message('user', turn['input']))
        remaining, rounds = turn.get('limit', 4096), 0
        responses = iter(turn['responses'])
        while remaining > 0:
            text = reference_render(history, True, thinking, tools if enabled else [], template)
            ids = tokenizer.encode(text).ids
            if len(ids) > turn.get('context', 8192):
                saw_warning = True
                break
            reused = next((i for i, (a, b) in enumerate(zip(cached, ids)) if a != b), min(len(cached), len(ids)))
            prompts.append(dict(text=text, ids=ids.copy(), reused=min(reused, len(ids)-1)))
            if show:
                visible += text + '\n'
            response = tokenizer.encode(next(responses)).ids + [eos]
            generated, complete = [], False
            # Host requests one token for prefill, then count-sized batches.
            first = True
            while response and remaining > 0 and len(ids) <= turn.get('context', 8192):
                amount = min(1 if first else turn.get('count', 5), remaining)
                amount = min(amount, turn.get('context', 8192)-len(ids)+1)
                first = False
                batch, response = response[:amount], response[amount:]
                ids += batch
                cached = ids[:-1]
                remaining -= len(batch)
                complete = batch[-1] == eos
                generated += batch[:-1] if complete else batch
                if complete:
                    break
            assistant, answer, warnings = parse_output(tokenizer, generated, complete)
            saw_warning |= warnings
            visible += tokenizer.decode(generated if show else answer, skip_special_tokens=False) + '\n'
            history.append(assistant)
            if not assistant['tool_calls']:
                break
            capped = rounds >= turn.get('rounds', 8)
            for call in assistant['tool_calls']:
                result = html_json({'error': 'Tool round limit reached'}) if capped else tool_result(server, enabled, call)
                assert result is not None
                if not capped:
                    executions.append(dict(call=call, result=result))
                history.append(message('tool', result))
            if capped:
                saw_warning = True
                break
            rounds += 1
        else:
            saw_warning = True
    return dict(messages=history, prompts=prompts, executions=executions, visible=visible), saw_warning


def fixtures():
    def call(name='sumN1', n=10, **extra):
        return '<tool_call>\n' + json.dumps(dict(name=name, arguments=dict(n=n, **extra))) + '\n</tool_call>'
    base = [
        dict(turns=[dict(input='Explain', thinking=True, responses=['<think>\nReason α🙂\n</think>\n\nAnswer']),
                    dict(input='Now no thinking', thinking=False, responses=['Short answer'])]),
        dict(messages=[message('system', 'Be precise <&>')], turns=[dict(input='Sum ten', tools=True, thinking=True,
             responses=['<think>Use the sum tool</think>' + call(), 'The result is 55.'])]),
        dict(turns=[dict(input='Both sums', tools=True, responses=[call() + call('sumN2', 10), '55 and 385'])]),
        dict(turns=[dict(input='Two steps', tools=True, thinking=True,
             responses=['<think>First step</think>' + call(n=3), '<think>Next step</think>Also ' + call('sumN2', 3), '6 and 14']),
                    dict(input='Next question', thinking=True, responses=['<think>New reasoning</think>Done'])]),
        dict(turns=[dict(input='Show everything', show=True, tools=True, thinking=True,
             responses=['<think>Reason</think>' + call(n=3), '<think>Finish</think>Result'])]),
        dict(turns=[dict(input='Bad syntax', tools=True, responses=['<tool_call>{bad}</tool_call>'])]),
        dict(turns=[dict(input='No close', tools=True, responses=['<tool_call>{"name":"sumN1"'])]),
        dict(turns=[dict(input='No thinking close', thinking=True, responses=['<think>unfinished'])]),
        dict(turns=[dict(input='Do not execute thoughts', tools=True, thinking=True,
             responses=['<think>' + call() + '</think>Answer'])]),
        dict(turns=[dict(input='Nested calls', tools=True,
             responses=['<tool_call>' + call() + '</tool_call>'])]),
        dict(turns=[dict(input='Unmatched close', tools=True, responses=['oops</tool_call>'])]),
        dict(turns=[dict(input='Budget', tools=True, limit=20, responses=[call()])]),
        dict(turns=[dict(input='Tools disabled', responses=[call(), 'Cannot use tools'])]),
        dict(turns=[dict(input='Stop looping', tools=True, rounds=1,
             responses=[call(n=2), call('sumN2', 2)])]),
        dict(turns=[dict(input='UTF8', count=1, thinking=True, responses=['🙂<think>α🙂</think>é🙂'])]),
        dict(turns=[dict(input='Context too small', tools=True, context=16, responses=[])]),
        dict(turns=[dict(input='Zero', tools=True, responses=[call(n=0) + call('sumN2', 0), 'Both zero'])]),
        dict(turns=[dict(input='Bounded maximum', tools=True, responses=[call('sumN2', 1000000), 'Computed'])]),
        dict(turns=[dict(input='Wrapped call', tools=True,
             responses=['<tool_call>{"function":{"name":"sumN1","arguments":{"n":4}}}</tool_call>', '10'])]),
        dict(turns=[dict(input='Duplicate key', tools=True,
             responses=['<tool_call>{"name":"sumN1","arguments":{"n":3,"n":4}}</tool_call>'])]),
        dict(turns=[dict(input='Null arguments', tools=True,
             responses=['<tool_call>{"name":"sumN1","arguments":null}</tool_call>'])]),
    ]
    for name, n in [('missing', 3), ('sumN1', -1), ('sumN2', 1000001), ('sumN1', 2**63), ('sumN1', '3'), ('sumN1', 1.5), ('sumN1', True)]:
        base.append(dict(turns=[dict(input='Handle error', tools=True, responses=[call(name, n), 'Recovered'])]))
    # Escaped JSON keys/names and strings in nested extra arguments are decoded
    # and safely reserialized, including surrogate-pair Unicode escapes.
    escaped = '<tool_call>{"na\\u006de":"su\\u006dN1","arguments":{"n":3,"note":"quote: \\\" <&> \\ud83d\\ude42"}}</tool_call>'
    base.append(dict(turns=[dict(input='Escapes', tools=True, responses=[escaped, '6'])]))
    string_args = '<tool_call>' + json.dumps(dict(name='sumN2', arguments='{"n": 3}')) + '</tool_call>'
    base.append(dict(turns=[dict(input='String args', tools=True, responses=[string_args, '14'])]))
    # Exercise each scenario with different generation chunk boundaries.
    return [dict(copy.deepcopy(test), turns=[dict(turn, count=count) for turn in copy.deepcopy(test['turns'])])
            for test in base for count in [1, 7]]


def check_render(exe, model, work, tools, template):
    histories = [
        [message('user', 'Hi'), message('assistant', 'Answer', 'Reason α\n')],
        [message('system', 'System <&>'), message('user', 'Hi'),
         message('assistant', '', 'Reason', [dict(name='sumN1', arguments={'n': 3})]), message('tool', '6')],
        [message('user', 'Hi'), message('assistant', 'Before', 'R1', [dict(name='sumN1', arguments={'n': 3})]),
         message('tool', '6'), message('tool', '14'), message('assistant', 'After', 'R2'), message('user', 'New')],
        [message('user', 'Hi'), message('assistant', '<think>\nLegacy\n</think>\nAnswer')],
        [message('user', '<tool_response>legacy</tool_response>'), message('assistant', 'Answer', '\nR\n')],
    ]
    cases = [dict(messages=h, generation=gen, thinking=think, tools=t)
             for h in histories for gen in [False, True] for think in [False, True] for t in [[], tools]]
    file = work / 'renders.jsonl'
    file.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    lines = parity.sml(exe, model, '--render-test', file).stdout.splitlines()
    assert len(lines) == len(cases)
    for case, actual in zip(cases, lines):
        expected = reference_render(case['messages'], case['generation'], case['thinking'], case['tools'], template)
        assert bytes.fromhex(actual).decode() == expected, (case, bytes.fromhex(actual).decode(), expected)
    print(f'{len(cases)} reasoning/tool chat renderings match the reference')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', default='./build/chat-sml')
    parser.add_argument('--session-exe', default='./build/session-test')
    parser.add_argument('--model')
    args = parser.parse_args()
    exe, driver = Path(args.exe).resolve(), Path(args.session_exe).resolve()
    with tempfile.TemporaryDirectory(prefix='fuchat-conversation-') as d:
        work = Path(d)
        model = Path(args.model).resolve() if args.model else work / 'tiny.gguf'
        if not args.model:
            parity.tiny_model(model)
        reader = GGUFReader(model)
        tokenizer = parity.chat.make_tokenizer(reader)
        eos = reader.get_field('tokenizer.ggml.eos_token_id').contents()
        server_path = work / 'server'
        subprocess.run(['futhark', 'c', '--server', 'qwen-f32.fut', '-o', str(server_path)], check=True)
        with futhark_server.Server(str(server_path)) as server:
            tools = []
            for entry in server.cmd('entry_points'):
                if entry.startswith('tool_'):
                    _, name, desc = entry.split('_', 2)
                    tools.append(dict(type='function', function=dict(name=name, description='calculate the ' + desc.replace('_', ' '),
                        parameters=dict(type='object', properties=dict(n=dict(type='number', description=''))))))
            template = Environment(loader=BaseLoader).from_string(reader.get_field('tokenizer.chat_template').contents()) if args.model else None
            check_render(exe, model, work, tools, template)
            cases = fixtures()
            path = work / 'sessions.jsonl'
            path.write_text(''.join(json.dumps(test) + '\n' for test in cases))
            p = subprocess.run([str(driver), str(model), str(path)], text=True, capture_output=True)
            assert p.returncode == 0, p.stderr
            lines = p.stdout.splitlines()
            assert len(lines) == len(cases), (len(lines), len(cases))
            for i, (test, line) in enumerate(zip(cases, lines)):
                actual = json.loads(line)
                expected, warnings = expected_session(test, tokenizer, eos, tools, server, template)
                assert bool(actual.pop('warnings')) == warnings, (i, test, actual)
                assert actual == expected, (i, test, actual, expected)
            print(f'{len(cases)} scripted thinking/tool sessions match Python and real Futhark tools')
        # Test the interactive toggles through the actual executable as well.
        p = parity.sml(exe, model, '--cs', 128, '--max-new-tokens', 3,
                       input='think\nshow\nnothink\nshow\nquit\n')
        assert 'Thinking mode enabled' in p.stdout and 'Thinking mode disabled' in p.stdout
    print('Conversation checks passed')

if __name__ == '__main__':
    main()
