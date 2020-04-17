# 实现voxel滤波，并加载数据集中的文件进行验证

import os
import open3d
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import time

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel分辨率
# 输出：
#     filtered_points：降采样之后的点云

def voxel_filter(point_cloud, leaf_size):

    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)

    D_x = np.floor((x_max - x_min) / leaf_size)
    D_y = np.floor((y_max - y_min) / leaf_size)
    D_z = np.floor((z_max - z_min) / leaf_size)
    # print(D_x, D_y, D_z)

    # 计算每个点所在的voxel
    h = np.floor((point_cloud - np.array([x_min, y_min, z_min])) / leaf_size)

    # base formula calc
    h = h[:, 0] + h[:, 1] * D_x + h[:, 2] * D_x * D_y

    dh = pd.DataFrame(columns=["value", "index"])
    dh["value"] = h
    dh['index'] = [x for x in range(point_cloud.shape[0])] # [10000 rows x 2 columns]
    dh = dh.drop_duplicates(['value']) # [338 rows x 2 columns] 去除重复的点
    points_index = dh['index'].to_numpy() # 保存index

    filtered_points = point_cloud[points_index, :] # 根据inedx查找点

    filtered_points = np.array(filtered_points, dtype=np.float64) # numpy to array

    return filtered_points


def draw_bbox(point_cloud):

    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)

    bb1 = [x_max, y_min, z_max]
    bb2 = [x_max, y_max, z_max]
    bb3 = [x_max, y_max, z_min]
    bb4 = [x_max, y_min, z_min]
    bb5 = [x_min, y_min, z_max]
    bb6 = [x_min, y_max, z_max]
    bb7 = [x_min, y_max, z_min]
    bb8 = [x_min, y_min, z_min]

    bbox_v = np.array([bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8], dtype=np.float32)
    bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0],
                  [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6],[3, 7]]  # 某条线由哪两个点构成

    colors = [[0, 0, 1] for _ in range(len(bbox_lines))]  # Default blue

    # 构建bbox
    bbox = open3d.LineSet()
    bbox.lines = open3d.Vector2iVector(bbox_lines)
    bbox.colors = open3d.Vector3dVector(colors)
    bbox.points = open3d.Vector3dVector(bbox_v)

    return bbox

def PCA(data, correlation=False, sort=True):

    data_mean = data - np.mean(data,axis=0) # n*3
    eigenvalues, eigenvectors = np.linalg.eig(np.dot(data_mean.T,data_mean)) # (3,n)*(n,3)
    if sort:
        sort = eigenvalues.argsort()[::-1] # 返回从大到小的索引
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]
    return eigenvalues, eigenvectors


def main():

    points = np.genfromtxt('./data/airplane_0001.txt', delimiter=',') # (10000,6)
    points = pd.DataFrame(points[:, 0:3])
    points.columns = ['x', 'y', 'z']
    point_cloud_pynt = PyntCloud(points)
    point_cloud_open3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # open3d.visualization.draw_geometries([point_cloud_open3d]) # 显示原始点云

    # 调用voxel滤波函数，实现滤波
    voxel_resolution = 0.05
    filtered_cloud = voxel_filter(np.array(point_cloud_pynt.points), voxel_resolution)
    print("Original points num：{num1}, After filtered points num: {num2}"
          .format(num1=np.array(point_cloud_pynt.points).shape[0], num2=len(filtered_cloud)))

    # point_cloud_open3d.points = open3d.utility.Vector3dVector(filtered_cloud)
    #
    # # 显示bbox和滤波后的点云
    # bbox = draw_bbox(np.array(point_cloud_pynt.points))
    # open3d.visualization.draw_geometries([point_cloud_open3d, bbox])

    points = np.array(point_cloud_pynt.points)  # shape:(10000, 3)

    # 用PCA分析点云主方向
    w, v = PCA(points)

    pca_v = np.array([[0, 0, 0], v[:, 0], v[:, 1], v[:, 2]], dtype=np.float32)
    pca_lines = [[0, 1], [0, 2], [0, 3]]  # 某条线由哪两个点构成
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # Default blue

    # 定义PCA的三条线
    pca_line = open3d.LineSet()
    pca_line.lines = open3d.Vector2iVector(pca_lines)
    pca_line.colors = open3d.Vector3dVector(colors)
    pca_line.points = open3d.Vector3dVector(pca_v)


    # 循环计算每个点的法向量
    pcd_tree = open3d.geometry.KDTreeFlann(point_cloud_open3d)  # 创建一个对象
    normals = []

    search_num = 20  # 搜索近邻点的数量
    start = time.time()
    for i in range(points.shape[0]):
        _, index, _ = pcd_tree.search_knn_vector_3d(points[i, :], search_num)
        set = points[index, :]  # shape (20,3)
        _, set_v = PCA(set)
        normals.append(set_v[:, 2])

    print('use time:', time.time() - start)
    normals = np.array(normals, dtype=np.float64)
    print(normals.shape)

    point_cloud_open3d.normals = open3d.utility.Vector3dVector(normals)

    bbox = draw_bbox(np.array(point_cloud_pynt.points))

    open3d.visualization.draw_geometries([point_cloud_open3d, pca_line, bbox])

if __name__ == '__main__':
    main()
