/// <reference types="mocha" />
import * as assert from 'assert';
import { suite, test } from 'mocha';
import { LlamaServer } from '../../llama-server';
import { Application } from '../../application';

suite('LlamaServer Test Suite', () => {
	const createServer = (splitInlineReasoningTags: boolean, showReasoning = false) =>
		new LlamaServer({
			configuration: {
				agent_split_inline_reasoning_tags: splitInlineReasoningTags,
				agent_show_reasoning: showReasoning,
			},
		} as unknown as Application);

	test('agent_show_reasoning alone enables inline tag splitting', () => {
		// Regression test: a user who only enables "show reasoning" (without
		// separately discovering agent_split_inline_reasoning_tags) should
		// still see reasoning from models that inline it via <think> tags.
		const server = createServer(false, true);
		const filtered = (server as any).filterThoughtFromMsgs([
			{ role: 'assistant', content: 'Plan <think>hidden</think> done' },
		]);

		assert.deepStrictEqual(filtered, [{ role: 'assistant', content: 'Plan  done' }]);
	});

	test('splits inline think tags across streamed chunks', () => {
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as (chunk: string) => {
			visible: string;
			reasoning: string;
		};

		let visible = '';
		let reasoning = '';

		for (const chunk of ['Hello <thi', 'nk>rea', 'soning</th', 'ink> world']) {
			const delta = splitInlineThoughts(chunk);
			visible += delta.visible;
			reasoning += delta.reasoning;
		}

		assert.strictEqual(visible, 'Hello  world');
		assert.strictEqual(reasoning, 'reasoning');
	});

	test('splits multiple inline reasoning blocks while preserving visible text', () => {
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as (chunk: string) => {
			visible: string;
			reasoning: string;
		};

		let visible = '';
		let reasoning = '';

		for (const chunk of ['Prefix <think>one</think> mid <think>two', '</think> suffix']) {
			const delta = splitInlineThoughts(chunk);
			visible += delta.visible;
			reasoning += delta.reasoning;
		}

		assert.strictEqual(visible, 'Prefix  mid  suffix');
		assert.strictEqual(reasoning, 'onetwo');
	});

	test('preserves literal think tags when inline splitting is disabled', () => {
		const server = createServer(false);
		const filtered = (server as any).filterThoughtFromMsgs([
			{
				role: 'assistant',
				content: 'Plan <think>hidden</think> done',
				reasoning_content: 'internal notes',
				name: 'assistant-one',
			},
		]);

		assert.deepStrictEqual(filtered, [
			{
				role: 'assistant',
				content: 'Plan <think>hidden</think> done',
				name: 'assistant-one',
			},
		]);
	});

	test('filters reasoning content out of assistant messages when enabled', () => {
		const server = createServer(true);
		const filtered = (server as any).filterThoughtFromMsgs([
			{
				role: 'assistant',
				content: 'Plan <think>hidden</think> done',
				reasoning_content: 'internal notes',
				name: 'assistant-one',
			},
		]);

		assert.deepStrictEqual(filtered, [
			{
				role: 'assistant',
				content: 'Plan  done',
				name: 'assistant-one',
			},
		]);
	});
});
