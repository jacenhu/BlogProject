import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import { mkdtemp, readFile, readdir, writeFile, rm, mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { test } from 'node:test'
import sharp from 'sharp'
import { createMarkdownRenderer } from 'vitepress'
import { catalogPlugin } from '../docs/.vitepress/catalog.js'
import { optimizeImages } from '../scripts/optimize-images.js'

const root = fileURLToPath(new URL('../', import.meta.url))

test('catalog renders every article and preserves heading anchors', async () => {
  const sidebar = JSON.parse(await readFile(new URL('../docs/.vitepress/sidebar.json', import.meta.url)))
  const groups = sidebar['/program/']
  const md = await createMarkdownRenderer(join(root, 'docs'), {
    config: (md) => md.use(catalogPlugin, groups)
  })
  const source = await readFile(new URL('../docs/program/index.md', import.meta.url), 'utf8')
  const html = md.render(source, { relativePath: 'program/index.md' })
  const paths = new Set()
  for (const group of groups) {
    for (const { link } of group.items) {
      assert.ok(!paths.has(link), `Duplicate article: ${link}`)
      paths.add(link)
      await readFile(join(root, 'docs', `${link}.md`))
    }
  }
  const articles = (await readdir(join(root, 'docs/program'), { recursive: true }))
    .filter((path) => path.endsWith('.md') && path !== 'index.md')
    .map((path) => `/program/${path.replaceAll('\\', '/').replace(/\.md$/, '')}`)
  assert.deepEqual([...paths].sort(), articles.sort())
  assert.equal((html.match(/<li>/g) || []).length, paths.size)
  for (const id of ['c', 'llm', 'kv-cache', 'java后端', '论文', 'ai-infra-专题']) {
    assert.ok(html.includes(`id="${id}"`), `Missing heading: ${id}`)
  }
  assert.ok(!html.includes('@catalog'))
  assert.ok(md.render(source, { relativePath: 'other.md' }).includes('@catalog'))
})

async function fixture(t) {
  const dir = await mkdtemp(join(tmpdir(), 'blog-images-'))
  t.after(() => rm(dir, { recursive: true, force: true }))
  return dir
}

test('image optimization recurses, preserves dimensions and never enlarges files', async (t) => {
  const dir = await fixture(t)
  await mkdir(join(dir, 'nested'))
  const files = ['test.png', 'nested/test.JPG']
  for (const name of files) {
    await sharp({ create: { width: 64, height: 48, channels: 3, background: '#0071e3' } })
      .toFile(join(dir, name))
  }
  await writeFile(join(dir, 'untouched.svg'), '<svg/>')
  const originals = await Promise.all(files.map((name) => readFile(join(dir, name))))
  const result = await optimizeImages(dir)
  assert.equal(result.count, 2)
  let saved = 0
  for (const [index, name] of files.entries()) {
    const output = await readFile(join(dir, name))
    assert.ok(output.length <= originals[index].length)
    saved += originals[index].length - output.length
    const { width, height } = await sharp(output).metadata()
    assert.deepEqual({ width, height }, { width: 64, height: 48 })
  }
  assert.equal(result.saved, saved)
  assert.equal(await readFile(join(dir, 'untouched.svg'), 'utf8'), '<svg/>')
  assert.ok(!(await readdir(dir, { recursive: true })).some((name) => name.endsWith('.tmp')))
})

test('corrupt images fail the CLI, preserve originals and leave no temporary files', async (t) => {
  const dir = await fixture(t)
  const original = 'invalid image'
  await writeFile(join(dir, 'broken.png'), original)
  const result = spawnSync(process.execPath, [join(root, 'scripts/optimize-images.js'), dir], { encoding: 'utf8' })
  assert.equal(result.status, 1)
  assert.match(result.stderr, /broken\.png/)
  assert.equal(await readFile(join(dir, 'broken.png'), 'utf8'), original)
  assert.deepEqual(await readdir(dir), ['broken.png'])
})
