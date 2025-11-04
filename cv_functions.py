# cv_functions.py
import cv2
import numpy as np
import glob

# --- 1. 相機校準函式 ---

def find_corners_from_images(image_paths):
    """
    在所有影像中尋找角點。
    """
    chessboard_size = (11, 8)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)

    # 準備 3D 座標 (0,0,0), (1,0,0), ...
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    objpoints = [] # 3D 點
    imgpoints = [] # 2D 點
    img_size = None
    images_with_corners = []

    for fname in image_paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img_size is None:
            img_size = gray.shape[::-1] # (w, h)

        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        if ret:
            objpoints.append(objp)
            corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners_subpix)

            # 繪製並儲存影像以供顯示
            cv2.drawChessboardCorners(img, chessboard_size, corners_subpix, ret)
            images_with_corners.append(img)
        else:
            print(f"在 {fname} 中未找到角點")

    print(f"角點偵測完成，共在 {len(imgpoints)} / {len(image_paths)} 張圖片中找到角點。")
    return objpoints, imgpoints, img_size, images_with_corners

def calibrate_camera(objpoints, imgpoints, img_size):
    """
    執行相機校準 (對應 1.2)
    """
    print("正在執行相機校準...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        img_size,
        None,
        None
    )
    if ret:
        print("相機校準成功。")
        return mtx, dist, rvecs, tvecs
    else:
        print("相機校準失敗。")
        return None, None, None, None

def get_extrinsic_matrix(rvec, tvec):
    """
    計算外參矩陣 (對應 1.3)
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    extrinsic_matrix = np.hstack((rotation_matrix, tvec))
    return extrinsic_matrix

def undistort_image(image_path, camera_matrix, dist_coeffs):
    """
    執行影像校正 (對應 1.5)
    """
    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, None
        
    undistorted_img = cv2.undistort(original_img, camera_matrix, dist_coeffs)
    return original_img, undistorted_img

# --- 2. 擴增實境函式 ---
def draw_ar_words(image, mtx, dist, rvec, tvec, word, fs, vertical):
    """
    在影像上繪製 AR 文字
    """
    # 字母在棋盤上的 6 個位置 (左上角座標)
    # (7,5), (4,5), (1,5)
    # (7,2), (4,2), (1,2)
    positions = [(7, 5, 0), (4, 5, 0), (1, 5, 0), (7, 2, 0), (4, 2, 0), (1, 2, 0)] 
    
    output_image = image.copy()
    
    for i, char in enumerate(word):
        if i >= len(positions):
            break
            
        # 取得字母的 3D 座標
        char_points_3d = fs.getNode(char.upper()).mat() 
        if char_points_3d is None:
            print(f"找不到字母 {char} 的座標")
            continue
            
        # 取得平移位置
        tx, ty, tz = positions[i]
        
        # 對字母的每個線段進行投影
        for line_segment in char_points_3d:
            # 每個 line_segment 包含 [p_start, p_end]
            # p_start 和 p_end 都是 3D 座標 (x, y, z)
            
            # 加上平移
            if vertical:
                # 垂直文字 (X, Y=0, Z)
                # P.14 範例: (x+7, 0+5, z)
                p1_world = [line_segment[0][0] + tx, 0 + ty, line_segment[0][2] + tz]
                p2_world = [line_segment[1][0] + tx, 0 + ty, line_segment[1][2] + tz]
            else:
                # 平面文字 (X, Y, Z=0)
                # P.13 範例: (x+7, y+5, 0)
                p1_world = [line_segment[0][0] + tx, line_segment[0][1] + ty, 0 + tz]
                p2_world = [line_segment[1][0] + tx, line_segment[1][1] + ty, 0 + tz]

            # 將 3D 世界座標點轉換為 2D 影像點
            points_to_project = np.array([p1_world, p2_world], dtype=np.float32)
            
            projected_points, _ = cv2.projectPoints(
                points_to_project, rvec, tvec, mtx, dist
            ) 
            
            # 轉換座標格式為 (x, y) 整數
            p1_img = (int(projected_points[0][0][0]), int(projected_points[0][0][1]))
            p2_img = (int(projected_points[1][0][0]), int(projected_points[1][0][1]))
            
            # 在影像上畫線
            cv2.line(output_image, p1_img, p2_img, (0, 0, 255), 3) 

    return output_image


# --- 3. 立體視差圖函式 ---
def compute_disparity_map(imgL_gray, imgR_gray):
    """
    計算並回傳正規化的視差圖
    """
    # 建立 StereoBM 物件
    numDisparities = 432
    blockSize = 25
    
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize) 
    
    # 計算視差
    disparity = stereo.compute(imgL_gray, imgR_gray)
    
    # 將視差圖正規化到 0-255 以便顯示 
    # disparity 的原始值是 16-bit, 需要轉成 8-bit
    disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    return disparity_normalized

# --- 4. SIFT 函式 ---
def detect_sift_keypoints(image):
    """
    偵測 SIFT 關鍵點並回傳 kp, des, 和繪製後的影像
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 建立 SIFT 物件
    sift = cv2.SIFT_create() 
    
    # 偵測關鍵點和描述子
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # 在影像上繪製關鍵點
    img_with_keypoints = cv2.drawKeypoints(
        gray, keypoints, None, color=(0, 255, 0), 
        flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    
    return keypoints, descriptors, img_with_keypoints

def match_sift_keypoints(img1, kp1, des1, img2, kp2, des2):
    """
    匹配 SIFT 關鍵點並回傳繪製後的匹配影像
    """
    # 建立 BFMatcher 
    bf = cv2.BFMatcher()
    
    # 使用 k-NN 匹配
    matches = bf.knnMatch(des1, des2, k=2) 
    
    # 套用 Ratio Test (Lowe's ratio test)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance: 
            good_matches.append(m)
            
    # 繪製匹配結果
    good_matches_for_drawing = [[m] for m in good_matches]

    img_matches = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, 
        good_matches_for_drawing, 
        None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    return img_matches

# --- 5. 影像轉換函式 ---
def transform_image(image, angle, scale, tx, ty):
    """
    執行 5.1 旋轉、缩放、平移
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # 取得 2x3 旋轉/縮放矩陣
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 加上平移
    M[0, 2] += tx
    M[1, 2] += ty
    
    # 套用仿射變換
    transformed = cv2.warpAffine(image, M, (w, h)) 
    
    return transformed

def perspective_transform(image):
    """
    執行 5.2 透視轉換
    """
    (h, w) = image.shape[:2]
    
    # 選擇影像的四個角
    src_points = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])
    
    # 將它們壓縮到中間
    dst_points = np.float32([
        [w * 0.2, h * 0.1],
        [w * 0.8, h * 0.1],
        [w * 0.1, h * 0.9],
        [w * 0.9, h * 0.9]
    ])
    
    # 計算透視變換矩陣
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 套用透視變換
    warped = cv2.warpPerspective(image, M, (w, h))
    
    return warped