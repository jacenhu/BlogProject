# Jacen's Blog

基于 VitePress 的技术博客，文章位于 `docs/program/`。

## 本地开发

使用 Node.js 22 或更高版本；CI 使用 `.nvmrc` 中的版本。

```sh
nvm use
npm ci
npm run dev
```

```sh
npm test        # 目录完整性、链接锚点、图片处理及失败退出检查
npm run build   # 生成静态页面并优化输出目录中的 PNG/JPEG
npm run preview # 预览构建结果
npm audit      # 检查包含开发依赖在内的安全告警
```

## 内容与主题

| 位置 | 用途 |
| --- | --- |
| `docs/program/**/*.md` | 技术文章，保留现有路径可避免旧链接失效 |
| `docs/.vitepress/sidebar.json` | 文章分类、顺序与标题的唯一目录配置 |
| `docs/program/index.md` | 技术首页，`<!-- @catalog -->` 在 Markdown 编译前展开为目录 |
| `docs/index.md` | 首页文案、专题卡片与精选文章数据 |
| `docs/.vitepress/nav.json` | 顶部导航 |
| `docs/.vitepress/theme/components/` | 首页布局与侧边栏折叠组件及其样式 |
| `docs/.vitepress/theme/style.css` | 全站主题样式 |
| `docs/public/` | 原始静态资源，构建不会改写这些文件 |

新增文章时，在 `docs/program/` 对应分类下创建 Markdown 文件，并在
`sidebar.json` 中添加一条链接；侧边栏与技术首页会同时更新。
目录以普通 Markdown 编译，因此保留页内锚点、右侧大纲和 VitePress 链接检查。
修改后运行 `npm test` 和 `npm run build`。

首页通过推理过程示意图链接相关笔记，文章与专题导航分别展示，支持键盘访问。
侧边栏折叠偏好保存在本地；移动端继续使用 VitePress 自带的目录按钮。

## 构建与部署

统一使用 `.github/workflows/deploy.yml`：PR 执行测试和构建；推送到 `master`
或在 `master` 手动运行工作流时，测试及构建通过后发布到
`jacenhu/jacenhu.github.io` 的 `master` 分支。
部署使用仓库已有的 `DEPLOY_TOKEN` secret；同一分支的运行串行执行。

构建产物位于 `docs/.vitepress/dist/`。图片优化只处理构建产物：压缩结果更小
才替换，遇到损坏图片或写入错误时返回失败并清理临时文件，阻止发布不完整产物。
旧的 Travis/VuePress 配置与本地强制推送脚本已移除。

## 依赖维护

提交 `package.json` 与 `package-lock.json`，CI 使用 `npm ci` 保证版本一致。
`overrides` 中保留了此前安全修复：Vite/esbuild 的版本约束，以及
`speech-rule-engine` 对 `@xmldom/xmldom` 的安全版本替换。
升级上游后，只有确认不再需要且 `npm audit`、测试、构建都通过，才移除约束。
