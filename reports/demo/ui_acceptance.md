# 真实浏览器界面验收

生成时间：2026-10-20T16:10:48.332403+00:00

使用真实 Streamlit、真实 API 与本地模型；未拦截、替换或模拟接口回复。浏览器禁用硬件 GPU。

- 通过：real_api_and_local_model_ready。
- 通过：initial_page。
- 未通过：search_feedback_comparison。 Locator.check: Timeout 15000ms exceeded.
Call log:
  - waiting for get_by_label("设置历史价格预算", exact=True)
    - locator resolved to <input title="" tabindex="0" role="switch" type="checkbox" aria-label="设置历史价格预算" data-react-aria-pressable="true"/>
  - attempting click action
    - waiting for element to be visible, enabled and stable
    - element is visible, enabled and stable
    - scrolling into view if needed
    - done scrolling
    - <label data-rac="" data-react-aria-pressable="true" class="st-emotion-cache-7tv5ne e1ap2qgk6">…</label> intercepts pointer events
  - retrying click action
    - waiting for element to be visible, enabled and stable
    - element is visible, enabled and stable
    - scrolling into view if needed
    - done scrolling
    - <div data-testid="stSidebarContent" class="st-emotion-cache-1csi29a e15rayve2">…</div> intercepts pointer events
  - retrying click action
    - waiting 20ms
    - waiting for element to be visible, enabled and stable
    - element is visible, enabled and stable
    - scrolling into view if needed
    - done scrolling
    - <label data-rac="" data-react-aria-pressable="true" class="st-emotion-cache-7tv5ne e1ap2qgk6">…</label> intercepts pointer events
  2 × retrying click action
      - waiting 100ms
      - waiting for element to be visible, enabled and stable
      - element is visible, enabled and stable
      - scrolling into view if needed
      - done scrolling
      - <div class="st-emotion-cache-1u2yxsq e1ap2qgk8"></div> from <div class="st-emotion-cache-1rm2k41 e1ap2qgk7">…</div> subtree intercepts pointer events

- 通过：chat_turn_1_response_rendered。
- 通过：chat_turn_1_budget。 {'prices_usd': [11.95, 21.67, 19.97, 13.99, 24.99], 'budget_usd': 35}
- 通过：chat_turn_2_response_rendered。
- 通过：chat_turn_2_budget。 {'prices_usd': [11.95, 19.97, 13.99, 15.92, 19.99], 'budget_usd': 20}
- 通过：no_browser_page_errors。

## 素材

- `reports/demo/01_home.png`
- `reports/demo/search_feedback_comparison_error.png`
- `reports/demo/04_chat_first_turn.png`
- `reports/demo/05_chat_budget_update.png`
- `reports/demo/choiceagent_demo.mp4`
- `reports/demo/original_video/page@17dbab3b689d0bee303734a513385d4c.webm`

## 浏览器错误

{
  "page_errors": [],
  "console_errors": []
}
