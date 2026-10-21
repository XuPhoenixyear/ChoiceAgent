# 真实浏览器界面验收

生成时间：2026-10-20T17:33:37.659291+00:00

使用真实 Streamlit、真实 API 与本地模型；未拦截、替换或模拟接口回复。浏览器禁用硬件 GPU。

对话请求组：original。完整请求与实际回复保留在 JSON 报告中。

- 通过：real_api_and_local_model_ready。
- 通过：initial_page。
- 通过：chinese_search_and_35_usd_budget。 {'query': '适合小桌面的收纳盒', 'prices_usd': [13.5, 24.99, 19.99, 21.67, 13.99]}
- 通过：product_titles_chinese。 ['REPUS Industries · 笔筒与文具收纳', 'Woodaholic turquoise 2层抽屉单元 - 欧洲国家风格桌面收纳盒 - 9.7x7x5.1" 青蓝色木制可堆叠抽屉 - 桌面收纳抽屉带标签架 - 双文件抽屉单元', 'RHCSZ · 桌面收纳架', 'Cretkhp 皮质办公文件夹带座垫，五隔储物盒，商务用品收纳箱，灰色条纹', 'BTSKY 皮质办公桌笔筒-多功能桌面收纳盒 商务卡笔铅笔移动电话遥控站配件家居办公桌装饰(黑色)']
- 通过：english_original_collapsed_by_default。
- 通过：english_original_available。
- 通过：chinese_category_submits_original_id。
- 通过：like_persisted_in_ui。
- 通过：save_control_and_product_title_chinese。
- 通过：compare_two_visible_products。
- 通过：chat_turn_1_answer_chinese。 你希望查找哪类商品、用于什么场景？
- 通过：chat_turn_1_response_rendered。
- 未通过：chat_turn_1_budget。 {'prices_usd': [], 'budget_usd': 35}
- 通过：chat_turn_1_tab_remains_active。
- 通过：chat_turn_2_answer_chinese。 按当前用途、预算和本次偏好筛选后，以下商品可供进一步比较：

EDENMS · 桌面收纳架（历史价格 $11.95）
Beiz Houseware 贝兹家居办公文件架及配件收纳，4层桌面纸 Tray 用于信件/ A4 文件，办公桌组织，办公室，宿舍收集办公用品（黑色）（历史价格 $19.97）
BTSKY 皮质办公桌笔筒-多功能桌面收纳盒 商务卡笔铅笔移动电话遥控站配件家居办公桌装饰(黑色)（历史价格 $13.99） 商品属性与价格可展开查看原始字段证据；这些不是实时售价。
- 通过：chat_turn_2_response_rendered。
- 通过：chat_turn_2_budget。 {'prices_usd': [11.95, 19.97, 13.99, 15.92, 19.99], 'budget_usd': 20}
- 通过：chat_turn_2_tab_remains_active。
- 通过：no_browser_page_errors。

## 素材

- `reports/localization/ui_final/01_home.png`
- `reports/localization/ui_final/02_search_budget.png`
- `reports/localization/ui_final/02b_english_original.png`
- `reports/localization/ui_final/03_comparison.png`
- `reports/localization/ui_final/04_chat_first_turn.png`
- `reports/localization/ui_final/05_chat_budget_update.png`
- `reports/localization/ui_final/choiceagent_demo.mp4`
- `reports/localization/ui_final/original_video/page@8887065e7f0e89d61fe872a2ff9284b9.webm`

## 浏览器错误

{
  "page_errors": [],
  "console_errors": []
}
