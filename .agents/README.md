# 项目 Skills

已按 ChoiceAgent 当前的 Python / FastAPI / Streamlit / SQLite 实现配置 20 个项目级技能。Codex 从 `.agents/skills/` 发现技能；这是本地 Skill 安装，不会安装 Claude 插件、全局钩子或改变模型配置。在此项目目录对应的 Codex 会话中开启下一轮使用；若未被发现，重新打开该项目会话或重启 Codex。

## 用途与来源

| 来源 | 数量 | 用途 |
| --- | --- | --- |
| [obra/superpowers](https://github.com/obra/superpowers) | 15 | 设计、计划、执行、测试、调试、审查、验证及配套维护技能 |
| [addyosmani/agent-skills](https://github.com/addyosmani/agent-skills) | 3 | `api-and-interface-design`、`documentation-and-adrs`、`security-and-hardening` |
| [anthropics/skills](https://github.com/anthropics/skills/tree/main/skills/webapp-testing) | 1 | `webapp-testing`：Playwright 浏览器检查与辅助脚本 |
| [nextlevelbuilder/ui-ux-pro-max-skill](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) | 1 | `ui-ux-pro-max`：交互、布局、视觉与可访问性资料查询 |

Superpowers 包含 `brainstorming`、`diagnosing-superpowers`、`dispatching-parallel-agents`、`executing-plans`、`finishing-a-development-branch`、`receiving-code-review`、`requesting-code-review`、`subagent-driven-development`、`systematic-debugging`、`test-driven-development`、`using-git-worktrees`、`using-superpowers`、`verification-before-completion`、`writing-plans` 和 `writing-skills`。保留整套技能以满足其同级引用，按任务读取，不同时启用全部流程。

未安装依赖 React/Next.js 的 Vercel 技能和依赖 Postgres 的 Supabase 技能。当前界面是 Streamlit，存储是 SQLite；未来技术栈变化后再增补。

## 使用

开发入口与任务路由见项目根目录 `AGENTS.md`。可以直接要求“先设计并写实施计划”“按计划实现并验证”“审查商品链接导入的接口与安全边界”或“检查手机端比较表和错误反馈”。代理应按任务读取相应技能，并遵守项目已有环境与业务约束。

UI 资料查询无需模型或额外依赖，例如在项目根目录执行：

```bash
PYTHONNOUSERSITE=1 /datas/wuxi/anaconda3/envs/my_rec/bin/python \
  .agents/skills/ui-ux-pro-max/scripts/search.py \
  "ecommerce" --domain product -n 1 --json
```

当前 UI 技能没有 Streamlit 专用栈，采用通用域建议。Playwright 已在现有项目环境中提供；浏览器测试复用项目浏览器目录和服务，无需为了安装技能启动服务。

## 版本与本地适配

`skills.lock.json` 记录四个来源的固定提交、技能路径、逐文件 SHA-256、上游 Git blob 和执行权限。下载文件与固定提交逐一校验，补齐所需配套资料。

- 恢复 GitHub ZIP 安装未保留的上游可执行权限。
- UI 技能文档中的 Claude 插件路径替换为本项目 Skill 的绝对路径。移动项目目录时应同步调整该路径并更新锁文件。
- Addy 安全技能引用的共享清单补放至 `references/security-checklist.md`。
- 根目录许可证存于 `licenses/`；`webapp-testing` 的许可证保留在其技能目录中。
- 不安装上游插件钩子，不修改用户全局设置。可选 Node/Graphviz 辅助功能不在本次安装验证范围；无需为常规开发额外安装。

验证结果见 `verification.json`。升级时重新审查上游变更、固定新提交、保留本地适配，并更新文件哈希与验证记录；不要直接覆盖当前技能。
