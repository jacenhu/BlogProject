// Expand the catalog before Markdown parsing so headings, anchors and link
// validation behave exactly like handwritten Markdown.
export function catalogPlugin(md, groups) {
  md.core.ruler.before('normalize', 'article-catalog', (state) => {
    if (state.env.relativePath === 'program/index.md') {
      state.src = state.src.replace('<!-- @catalog -->', renderCatalog(groups))
    }
  })
}

export function renderCatalog(groups) {
  return groups.map(({ text, items }) => [
    `## ${text}`,
    '',
    ...items.map(({ text, link }) => `- [${text.replace(/([\[\]])/g, '\\$1')}](${link})`)
  ].join('\n')).join('\n\n')
}
