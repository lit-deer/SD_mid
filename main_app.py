# main_app.py
import sys
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QApplication, QFileDialog
import cv2
import numpy as np
import glob

# 匯入我們將要編寫的 CV 函式庫
import cv_functions



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        # 載入 .ui 檔案
        uic.loadUi('MainWindow-cvdlhw1.ui', self)
        
        # --- 初始化變數 ---
        # self.calibrator = None      # 用於儲存校準物件
        self.root_path = ""
        self.calib_image_folder = "" # 校準影像資料夾
        self.calib_images_paths = [] # 儲存校準圖片的路徑
        self.objpoints = []          # 3D 點
        self.imgpoints = []          # 2D 點
        self.img_size = None         # 影像大小
        self.camera_matrix = None    # 內參 K
        self.dist_coeffs = None      # 畸變 D
        self.rvecs = None            # 旋轉 R
        self.tvecs = None            # 平移 T
        self.stereo_imgL = None      # 3.1 左影像
        self.stereo_imgR = None      # 3.1 右影像
        self.sift_img1 = None        # 4.1 影像1
        self.sift_img2 = None        # 4.2 影像2
        self.sift_kp1 = None         # SIFT 影像1 的 Keypoints
        self.sift_des1 = None        # SIFT 影像1 的 Descriptors
        self.sift_kp2 = None         # SIFT 影像2 的 Keypoints
        self.sift_des2 = None        # SIFT 影像2 的 Descriptors

        # --- 連接按鈕 ---        
        # Load Image
        self.loadFolderButton.clicked.connect(self.on_load_folder)
        self.loadImageLButton.clicked.connect(self.on_load_image_L)
        self.loadImageRButton.clicked.connect(self.on_load_image_R)
        
        # 1. Calibration
        self.findCornersButton.clicked.connect(self.on_find_corners)
        self.findIntrinsicButton.clicked.connect(self.on_find_intrinsic)
        self.findExtrinsicButton.clicked.connect(self.on_find_extrinsic)
        self.findDistortionButton.clicked.connect(self.on_find_distortion)
        self.showResultButton.clicked.connect(self.on_show_result)

        # 2. Augmented Reality
        self.showWordsOnBoardButton.clicked.connect(self.on_show_words_on_board)
        self.showWordsVerticalButton.clicked.connect(self.on_show_words_vertical)

        # 3. Stereo Disparity Map
        self.stereoDisparityMapButton.clicked.connect(self.on_stereo_disparity_map)

        # 4. SIFT
        self.loadSiftImage1Button.clicked.connect(self.on_load_sift_image1)
        self.loadSiftImage2Button.clicked.connect(self.on_load_sift_image2)
        self.keypointsButton.clicked.connect(self.on_sift_keypoints)
        self.matchedKeypointsButton.clicked.connect(self.on_sift_matched_keypoints)

        self.show()
        print("UI 載入完成, 等待操作...")

    # --- 影像載入函式 ---
    def on_load_folder(self):
        # 載入資料夾
        folder = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if folder:
            self.calib_image_folder = folder
            # 讀取所有 .bmp 影像路徑
            self.calib_images_paths = sorted(glob.glob(f'{self.calib_image_folder}/*.bmp'))
            # self.calibrator = CameraCalibrator(self.calib_image_folder) # <--- 刪除這一行
            print(f"校準影像資料夾: {folder} 已載入。")
            print(f"找到 {len(self.calib_images_paths)} 張 .bmp 影像。")
            print("請點擊 '1.1 Find Corners'。")

    def on_load_image_L(self):
        # 載入左影像
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.png *.jpg *.bmp)")
        if fname:
            self.stereo_imgL = cv2.imread(fname)
            # 視差運算需要灰階影像
            self.stereo_imgL_gray = cv2.cvtColor(self.stereo_imgL, cv2.COLOR_BGR2GRAY)
            print("影像 Image_L 已載入。")

    def on_load_image_R(self):
        # 載入右影像
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.png *.jpg *.bmp)")
        if fname:
            self.stereo_imgR = cv2.imread(fname)
            # 視差運算需要灰階影像
            self.stereo_imgR_gray = cv2.cvtColor(self.stereo_imgR, cv2.COLOR_BGR2GRAY)
            print("影像 Image_R 已載入。")

    def on_load_sift_image1(self):
        # 載入 SIFT 影像 1
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if fname:
            self.sift_img1 = cv2.imread(fname)
            print("SIFT 影像 1 (Left.jpg) 已載入。")

    def on_load_sift_image2(self):
        # 載入 SIFT 影像 2
        fname, _ = QFileDialog.getOpenFileName(self, 'Open file', '', "Image files (*.jpg *.png)")
        if fname:
            self.sift_img2 = cv2.imread(fname)
            print("SIFT 影像 2 (Right.jpg) 已載入。")

    # --- 1. 相機校準函式 ---
    
    def on_find_corners(self):
        if not self.calib_images_paths: # 檢查路徑列表
            print("請先載入校準影像資料夾。")
            return
            
        # 呼叫 cv_functions 中的函式
        objp, imgp, size, images_with_corners = cv_functions.find_corners_from_images(
            self.calib_images_paths
        )
        
        # 儲存狀態
        self.objpoints = objp
        self.imgpoints = imgp
        self.img_size = size
        
        # 顯示結果
        print("開始顯示角點影像...")
        for img in images_with_corners:
            img_display = cv2.resize(img, (800, 800))
            cv2.imshow('1.1 Find Corners', img_display)
            cv2.waitKey(500)
        cv2.destroyAllWindows()
        print("角點顯示完畢。")

    def on_find_intrinsic(self):
        if not self.imgpoints: # 檢查 imgpoints 是否已填充
            print("請先執行 1.1 Find Corners。")
            return
            
        # 呼叫 cv_functions 中的函式
        mtx, dist, rvecs, tvecs = cv_functions.calibrate_camera(
            self.objpoints, self.imgpoints, self.img_size
        )
        
        if mtx is not None:
            # 儲存狀態
            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.rvecs = rvecs
            self.tvecs = tvecs
            
            print("--- 1.2 內參矩陣 (Intrinsic Matrix) ---")
            print(self.camera_matrix)
        else:
            print("校準失敗。")

    def on_find_extrinsic(self):
        if self.camera_matrix is None: # 檢查校準是否已完成
            print("請先執行 1.2 Find Intrinsic。")
            return
            
        selected_index = self.extrinsicSpinBox.value() - 1
        
        if selected_index >= len(self.rvecs):
            print(f"錯誤：索引 {selected_index} 超出範圍。")
            return

        # 呼叫 cv_functions 中的函式
        matrix = cv_functions.get_extrinsic_matrix(
            self.rvecs[selected_index], self.tvecs[selected_index]
        )
        
        if matrix is not None:
            print(f"--- 1.3 外參矩陣 (Extrinsic Matrix) for image {selected_index + 1} ---")
            print(matrix)

    def on_find_distortion(self):
        if self.dist_coeffs is None: # 檢查校準是否已完成
            print("請先執行 1.2 Find Intrinsic。")
            return
            
        print("--- 1.4 畸變矩陣 (Distortion Matrix) ---")
        print(self.dist_coeffs) # 直接從 self 讀取

    def on_show_result(self):
        if self.camera_matrix is None: # 檢查校準是否已完成
            print("請先執行 1.2 Find Intrinsic。")
            return
        
        selected_index = self.extrinsicSpinBox.value() - 1
        
        if selected_index >= len(self.calib_images_paths):
            print(f"錯誤：索引 {selected_index} 超出範圍。")
            return

        # 呼叫 cv_functions 中的函式
        original, undistorted = cv_functions.undistort_image(
            self.calib_images_paths[selected_index],
            self.camera_matrix,
            self.dist_coeffs
        )
        
        if original is not None:
            original_display = cv2.resize(original, (800, 800))
            undistorted_display = cv2.resize(undistorted, (800, 800))
            cv2.putText(original_display, 'Distorted', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(undistorted_display, 'Undistorted', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            comparison = np.hstack((original_display, undistorted_display))
            cv2.imshow(f'1.5 Show Result (Image {selected_index + 1})', comparison)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    # --- 2. 擴增實境 ---
    def on_show_words_on_board(self):
        self.run_augmented_reality(vertical=False) 

    def on_show_words_vertical(self):
        self.run_augmented_reality(vertical=True)

    def run_augmented_reality(self, vertical):
        if not self.root_path:
            print("請先載入主資料集 (點擊 Load Folder)。")
            return
            
        word = self.arTextBox.text()
        if len(word) > 6:
            print("錯誤：文字必須少於 6 個字元。")
            return
        
        if vertical:
            db_path = self.q2_db_vertical_path
            print("執行 2.2 Show Words Vertically")
        else:
            db_path = self.q2_db_onboard_path
            print("執行 2.1 Show Words on Board")

        print("為 Q2 (AR) 執行獨立的相機校準...")
        if not self.q2_image_paths:
            print("錯誤: Q2 影像路徑未載入。")
            return

        # 1. 找到 Q2 影像的角點
        q2_objpoints, q2_imgpoints, q2_img_size, _ = cv_functions.find_corners_from_images(self.q2_image_paths)
        
        if not q2_imgpoints:
            print("錯誤: 在 Q2 影像中找不到角點，無法執行 AR。")
            return

        # 2. 校準 Q2 影像 (取得 q2_mtx, q2_dist 等...)
        q2_mtx, q2_dist, q2_rvecs, q2_tvecs = cv_functions.calibrate_camera(
            q2_objpoints, q2_imgpoints, q2_img_size
        )
        
        if q2_mtx is None:
            print("錯誤: Q2 影像校準失敗，無法執行 AR。")
            return
        
        print("Q2 校準成功。開始繪製 AR...")

        fs = cv2.FileStorage(db_path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            print(f"錯誤: 無法開啟資料庫檔案 {db_path}")
            return
        
        # 迭代 Q2 的影像
        num_ar_images = min(len(self.q2_image_paths), len(q2_rvecs))
        for i in range(num_ar_images):
            img = cv2.imread(self.q2_image_paths[i])
            rvec = q2_rvecs[i]
            tvec = q2_tvecs[i]
            
            # 使用 Q2 獨立校準的參數 (q2_mtx, q2_dist)
            img_with_words = cv_functions.draw_ar_words(
                img, q2_mtx, q2_dist, rvec, tvec, word, fs, vertical
            )
            
            img_display = cv2.resize(img_with_words, (800, 800))
            cv2.imshow('Augmented Reality', img_display)
            cv2.waitKey(1000) # 顯示 1 秒

        cv2.destroyAllWindows()
        fs.release()

    # --- 3. 立體視差圖 ---
    def on_stereo_disparity_map(self):
        if self.stereo_imgL_gray is None or self.stereo_imgR_gray is None:
            print("請先載入 Image_L 和 Image_R。")
            return
        
        print("正在計算視差圖...")
        disparity_map = cv_functions.compute_disparity_map(
            self.stereo_imgL_gray, self.stereo_imgR_gray
        )
        cv2.imshow('ImgL', cv2.resize(self.stereo_imgL, (800, 600)))
        cv2.imshow('ImgR', cv2.resize(self.stereo_imgR, (800, 600)))
        cv2.imshow('3.1 Disparity Map', disparity_map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- 4. SIFT ---
    def on_sift_keypoints(self):
        if self.sift_img1 is None:
            print("請先載入 SIFT 影像 1 (Load Image 1)。")
            return
        
        print("正在偵測 SIFT 關鍵點...")
        # 儲存 kp 和 des 供 4.2 使用
        self.sift_kp1, self.sift_des1, img_with_kp = cv_functions.detect_sift_keypoints(
            self.sift_img1
        )
        
        cv2.imshow('4.1 SIFT Keypoints', img_with_kp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_sift_matched_keypoints(self):
        if self.sift_img1 is None or self.sift_img2 is None:
            print("請先載入 SIFT 影像 1 和 2。")
            return
            
        # 確保兩張圖的 kp 和 des 都已偵測
        if self.sift_kp1 is None:
            self.sift_kp1, self.sift_des1, _ = cv_functions.detect_sift_keypoints(self.sift_img1)
            
        if self.sift_kp2 is None:
            self.sift_kp2, self.sift_des2, _ = cv_functions.detect_sift_keypoints(self.sift_img2)

        print("正在匹配 SIFT 關鍵點...")
        img_matches = cv_functions.match_sift_keypoints(
            self.sift_img1, self.sift_kp1, self.sift_des1,
            self.sift_img2, self.sift_kp2, self.sift_des2
        )
        
        cv2.imshow('4.2 Matched SIFT Keypoints', img_matches)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # --- 5. 影像轉換 ---
    def on_rotation_scaling_translation(self):
        if self.sift_img1 is None:
            print("請先載入 SIFT 影像 1 (Load Image 1) 來進行轉換。")
            return
            
        try:
            angle = float(self.angleLineEdit.text())
            scale = float(self.scaleLineEdit.text())
            tx = float(self.txLineEdit.text())
            ty = float(self.tyLineEdit.text())
        except ValueError:
            print("錯誤：Angle, Scale, Tx, Ty 必須是數字。")
            return
            
        print(f"執行 5.1 轉換: Angle={angle}, Scale={scale}, Tx={tx}, Ty={ty}")
        transformed_img = cv_functions.transform_image(
            self.sift_img1, angle, scale, tx, ty
        )
        
        cv2.imshow('5.1 Rotation, Scaling, Translation', transformed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_perspective_transform(self):
        if self.sift_img1 is None:
            print("請先載入 SIFT 影像 1 (Load Image 1) 來進行轉換。")
            return
        
        print("執行 5.2 透視轉換...")
        warped_img = cv_functions.perspective_transform(self.sift_img1)
        
        cv2.imshow('5.2 Perspective Transform', warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())