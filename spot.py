import os
import cv2
import dlib
import numpy as np

# 检查并初始化Dlib的面部检测器和关键点检测器
predictor_path = "D:\Face fusion\shape_predictor_68_face_landmarks.dat\shape_predictor_68_face_landmarks.dat"
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"缺少预测器文件: {predictor_path}。请确保文件路径正确并且文件已下载。")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


# 从图像中检测关键点并添加更多点
def get_face_landmarks_with_background(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件未找到: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # 基本关键点
    points = []
    if len(faces) > 0:
        landmarks = predictor(gray, faces[0])
        points = [(p.x, p.y) for p in landmarks.parts()]

    # 获取图像尺寸
    h, w = img.shape[:2]

    # 添加四周的背景控制点
    margin = 10  # 距离边缘的距离
    border_points = [
        (margin, margin), (w // 2, margin), (w - margin, margin),  # 上边缘
        (margin, h // 2), (w - margin, h // 2),  # 左右中点
        (margin, h - margin), (w // 2, h - margin), (w - margin, h - margin)  # 下边缘
    ]

    # 顶部扩展点（覆盖头发区域）
    top_points = [(w // 2, margin * 3), (w // 4, margin * 3), (3 * w // 4, margin * 3)]

    # 底部扩展点（覆盖脖子和肩膀）
    bottom_points = [(w // 2, h - margin * 3), (w // 4, h - margin * 2), (3 * w // 4, h - margin * 2)]

    # 汇总所有点
    points.extend(border_points + top_points + bottom_points)

    return points


# 写入关键点文件
def write_points_to_file(points, file_path):
    with open(file_path, "w") as file:
        for (x, y) in points:
            file.write(f"{x} {y}\n")


# 生成三角剖分
def get_delaunay_triangles(points, img_size):
    rect = (0, 0, img_size[1], img_size[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangles = subdiv.getTriangleList()
    indices = []
    for t in triangles:
        pts = [(int(t[i]), int(t[i + 1])) for i in range(0, 6, 2)]
        idx = [points.index(pt) if pt in points else -1 for pt in pts]
        if -1 not in idx:
            indices.append(idx)
    return indices


# 将三角形索引写入文件
def write_triangles_to_file(triangles, file_path):
    with open(file_path, "w") as file:
        for tri in triangles:
            file.write(f"{tri[0]} {tri[1]} {tri[2]}\n")


if __name__ == "__main__":
    image_path1 = r"D:\Face fusion\5.png"
    image_path2 = r"D:\Face fusion\2.png"
    points_file1 = "1_points.txt"
    points_file2 = "2_points.txt"
    triangles_file = "triangles.txt"

    # 获取两张图片的关键点
    points1 = get_face_landmarks_with_background(image_path1)
    points2 = get_face_landmarks_with_background(image_path2)

    if points1 and points2:
        # 写入关键点文件
        write_points_to_file(points1, points_file1)
        write_points_to_file(points2, points_file2)

        # 生成并写入三角形文件
        img = cv2.imread(image_path1)
        triangles = get_delaunay_triangles(points1, img.shape)
        write_triangles_to_file(triangles, triangles_file)

        print("点文件和三角形文件生成成功！")
    else:
        print("关键点检测失败。")
