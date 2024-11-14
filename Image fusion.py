import cv2
import numpy as np
import sys

# 从文本文件读取点
def readPoints(path):
    points = []
    try:
        with open(path) as file:
            for line in file:
                x, y = line.split()
                points.append((int(x), int(y)))
    except FileNotFoundError:
        print(f"错误：未找到文件 {path}")
        sys.exit(1)
    return points

# 对src应用由srcTri和dstTri计算得到的仿射变换
def applyAffineTransform(src, srcTri, dstTri, size):
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return dst

# 将img1和img2的三角区域变形并进行alpha混合到img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # 获取三角形的边界矩形
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # 将三角形点坐标偏移至边界矩形的左上角
    t1Rect, t2Rect, tRect = [], [], []
    for i in range(3):
        tRect.append((t[i][0] - r[0], t[i][1] - r[1]))
        t1Rect.append((t1[i][0] - r1[0], t1[i][1] - r1[1]))
        t2Rect.append((t2[i][0] - r2[0], t2[i][1] - r2[1]))

    # 创建并填充三角形的掩码
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0)

    # 将图像小矩形区域应用变形
    img1Rect = img1[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    img2Rect = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    size = (r[2], r[3])

    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # 对矩形区域进行alpha混合
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # 将矩形区域的三角部分复制到输出图像
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = \
        img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask

if __name__ == '__main__':
    filename1 =  r"D:\Face fusion\5.png"
    filename2 =  r"D:\Face fusion\2.png"
    points_txt1 = "D:/Face fusion/1_points.txt"
    points_txt2 = "D:/Face fusion/2_points.txt"
    alpha = 0.5

    # 读取图像
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)

    if img1 is None or img2 is None:
        print("错误：图像文件未找到。")
        sys.exit(1)

    img1 = np.float32(img1)
    img2 = np.float32(img2)

    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 读取对应点
    points1 = readPoints(points_txt1)
    points2 = readPoints(points_txt2)
    points = [(int((1 - alpha) * p1[0] + alpha * p2[0]), int((1 - alpha) * p1[1] + alpha * p2[1]))
              for p1, p2 in zip(points1, points2)]

    imgMorph = np.zeros(img1.shape, dtype=img1.dtype)

    # 读取三角形数据
    with open("D:/Face fusion/triangles.txt") as file:
        for line in file:
            x, y, z = map(int, line.split())
            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    # 显示结果
    out_img = np.hstack((img1, imgMorph, img2))
    cv2.imshow("Morphed Face with Hair and Neck", np.uint8(imgMorph))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
