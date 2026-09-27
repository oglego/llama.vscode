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

	type Splitter = {
		push: (chunk: string) => { visible: string; reasoning: string };
		flush: () => { visible: string; reasoning: string };
	};

	const runSplitter = (splitter: Splitter, chunks: string[]) => {
		let visible = '';
		let reasoning = '';
		for (const chunk of chunks) {
			const delta = splitter.push(chunk);
			visible += delta.visible;
			reasoning += delta.reasoning;
		}
		const tail = splitter.flush();
		visible += tail.visible;
		reasoning += tail.reasoning;
		return { visible, reasoning };
	};

	test('splits inline think tags across streamed chunks', () => {
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as Splitter;

		const { visible, reasoning } = runSplitter(splitInlineThoughts, ['Hello <thi', 'nk>rea', 'soning</th', 'ink> world']);

		assert.strictEqual(visible, 'Hello  world');
		assert.strictEqual(reasoning, 'reasoning');
	});

	test('splits multiple inline reasoning blocks while preserving visible text', () => {
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as Splitter;

		const { visible, reasoning } = runSplitter(splitInlineThoughts, ['Prefix <think>one</think> mid <think>two', '</think> suffix']);

		assert.strictEqual(visible, 'Prefix  mid  suffix');
		assert.strictEqual(reasoning, 'onetwo');
	});

	test('flush emits a held-back partial tag prefix as visible text', () => {
		// Regression test: a stream that ends right after "<" or "<th" (not
		// yet a full opening tag) used to lose that text entirely, since
		// nothing ever emitted the splitter's internal buffer.
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as Splitter;

		const { visible, reasoning } = runSplitter(splitInlineThoughts, ['if a value < th', 'reshold']);

		assert.strictEqual(visible, 'if a value < threshold');
		assert.strictEqual(reasoning, '');
	});

	test('flush emits an unclosed think block as reasoning instead of dropping it', () => {
		// e.g. finish_reason=length cut the model off mid-thought.
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as Splitter;

		const { visible, reasoning } = runSplitter(splitInlineThoughts, ['<think>still working through it']);

		assert.strictEqual(visible, '');
		assert.strictEqual(reasoning, 'still working through it');
	});

	test('flush is a no-op once the buffer has already been drained', () => {
		const server = createServer(true);
		const splitInlineThoughts = (server as any).createInlineThoughtSplitter() as Splitter;

		splitInlineThoughts.push('<think>done</think>answer');
		const tail = splitInlineThoughts.flush();

		assert.strictEqual(tail.visible, '');
		assert.strictEqual(tail.reasoning, '');
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

	test('caps reasoning_content accumulated from inline <think> tags, even without agent_show_reasoning', () => {
		// Regression test: previously fullReasoning had no cap at all, so a
		// model reasoning at length would grow message.reasoning_content
		// without bound - and that message gets persisted into chat history
		// and re-sent as part of every future prompt.
		const Utils = require('../../utils').Utils;
		let fullReasoning = '';
		const splitter = (new LlamaServer({
			configuration: { agent_split_inline_reasoning_tags: true, agent_show_reasoning: false },
		} as any) as any).createInlineThoughtSplitter() as {
			push: (c: string) => { visible: string; reasoning: string };
			flush: () => { visible: string; reasoning: string };
		};

		const { reasoning } = splitter.push('<think>' + 'x'.repeat(25000) + '</think>done');
		fullReasoning = Utils.appendBounded(fullReasoning, reasoning, 20000, '[reasoning truncated]\n');

		assert.ok(fullReasoning.startsWith('[reasoning truncated]\n'));
		assert.ok(fullReasoning.length <= 20000);
	});
});
