# SD_mid
軟體設計期中專案

## 專案簡介
本專案為「電腦視覺與深度學習」的作業，專案內容為實作四項主要的電腦視覺任務，包含：相機校準 (Camera Calibration)、擴增實境 (Augmented Reality)、立體視差圖 (Stereo Disparity Map)、SIFT 特徵點配對。
整個專案分成三個主要檔案: main_app.py, cv_functions.py, MainWindow-cvdlhw1.ui。
1.	main_app.py負責作為專案的控制中心
-	載入 .ui 檔案來建立視覺介面。
-	綁定所有按鈕的點擊事件（e.g. findCornersButton.clicked.connect）。
-	管理應用程式的狀態，例如 self 變數中儲存相機矩陣、載入的影像等。
-	當使用者操作介面時，它負責呼叫 cv_functions.py 中對應的函式來執行實際運算，並將運算結果呈現給使用者。
2.	cv_functions.py 負責實現核心演算法
-	它包含了所有實際的電腦視覺運算函式，例如 find_corners_from_images、calibrate_camera、detect_and_match_sift、transform_image 等。
3.	MainWindow-cvdlhw1.ui負責組成專案的UI介面
-	它定義了所有視窗、按鈕、輸入框和標籤的佈局與外觀，main_app.py 會在執行時讀取此檔案來建立使用者介面。
