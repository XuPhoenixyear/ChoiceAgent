# 真实浏览器界面验收

生成时间：2026-10-20T16:15:56.863428+00:00

使用真实 Streamlit、真实 API 与本地模型；未拦截、替换或模拟接口回复。浏览器禁用硬件 GPU。

- 通过：real_api_and_local_model_ready。
- 通过：initial_page。
- 通过：chinese_search_and_35_usd_budget。 {'query': '适合小桌面的收纳盒', 'prices_usd': [13.5, 24.99, 19.99, 21.67, 13.99]}
- 通过：like_persisted_in_ui。
- 通过：compare_two_visible_products。
- 通过：chat_turn_1_response_rendered。
- 通过：chat_turn_1_budget。 {'prices_usd': [11.95, 21.67, 13.99, 15.92, 18.45], 'budget_usd': 35}
- 通过：chat_turn_1_tab_remains_active。
- 通过：chat_turn_2_response_rendered。
- 通过：chat_turn_2_budget。 {'prices_usd': [11.95, 13.99, 15.92, 18.45, 19.99], 'budget_usd': 20}
- 通过：chat_turn_2_tab_remains_active。
- 通过：no_browser_page_errors。

## 素材

- `reports/demo/final/01_home.png`
- `reports/demo/final/02_search_budget.png`
- `reports/demo/final/03_comparison.png`
- `reports/demo/final/04_chat_first_turn.png`
- `reports/demo/final/05_chat_budget_update.png`
- `reports/demo/final/choiceagent_demo.mp4`
- `reports/demo/final/original_video/page@8247f7ee183c5f540de8feeeb8b7688a.webm`

## 浏览器错误

{
  "page_errors": [],
  "console_errors": []
}
