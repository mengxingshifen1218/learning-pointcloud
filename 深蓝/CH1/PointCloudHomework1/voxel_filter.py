# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d
import random
import numpy as np
import time

def random_voxel_filter(point_cloud, leaf_size):

    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)

    D_x = np.floor((x_max - x_min) / leaf_size)
    D_y = np.floor((y_max - y_min) / leaf_size)
    D_z = np.floor((z_max - z_min) / leaf_size)

    # 计算每个点所在的voxel
    h = np.floor((point_cloud - np.array([x_min, y_min, z_min])) / leaf_size)

    # base formula calc
    h = h[:, 0] + h[:, 1] * D_x + h[:, 2] * D_x * D_y

    # 从小到大对h排序，输出索引
    index = np.argsort(h).tolist()
    h = h[index].tolist()

    # 构建dict保存key和value
    dict = {}
    for i, j in zip(h, index):
        if i not in dict.keys():
            dict[i] = j
        else:
            if isinstance(dict[i], list) == False:
                dict[i] = [dict[i]]
                dict[i].append(j)
            else:
                dict[i].append(j)

    filtered_points_index = []

    # 随机选择的index
    for item in dict.items():
        if isinstance(item[1], list) == True:
            filtered_points_index.append(random.choice(item[1]))
        else:
            filtered_points_index.append(item[1])

    filtered_points = point_cloud[filtered_points_index]
    filtered_points = np.array(filtered_points, dtype=np.float64)  # numpy to array

    return filtered_points


def centroid_voxel_filter(point_cloud, leaf_size):

    x_max, y_max, z_max = np.max(point_cloud, axis=0)
    x_min, y_min, z_min = np.min(point_cloud, axis=0)

    D_x = np.floor((x_max - x_min) / leaf_size)
    D_y = np.floor((y_max - y_min) / leaf_size)
    D_z = np.floor((z_max - z_min) / leaf_size)

    h = np.floor((point_cloud - np.array([x_min, y_min, z_min])) / leaf_size)

    h = h[:, 0] + h[:, 1] * D_x + h[:, 2] * D_x * D_y

    index = np.argsort(h).tolist()
    h = h[index].tolist()

    dict = {}
    for i, j in zip(h, index):
        if i not in dict.keys():
            dict[i] = j
        else:
            if isinstance(dict[i], list) == False:
                dict[i] = [dict[i]]
                dict[i].append(j)
            else:
                dict[i].append(j)

    filtered_points = []

    for item in dict.items():
        if isinstance(item[1], list) == True:
            avg_points_ = np.mean(point_cloud[item[1]], axis=0)
            filtered_points.append(avg_points_.tolist())
        else:
            filtered_points.append(point_cloud[item[1]])

    return filtered_points

def main():

    # 读取点云数据并显示
    init_points = np.loadtxt('./data/airplane_0001.txt', delimiter=',').astype(np.float32)[:, 0:3]
    pointcloud = open3d.PointCloud()
    pointcloud.points = open3d.utility.Vector3dVector(init_points)
    # open3d.visualization.draw_geometries([pointcloud],window_name='init_pointcloud')

    # 调用voxel滤波函数，实现滤波
    # 使用中心点
    voxel_resolution = 0.01
    start = time.time()
    # filtered_cloud = centroid_voxel_filter(init_points, voxel_resolution)
    print('centroid_voxel_filter use time:', time.time() - start)

    # 使用随机采样
    start = time.time()
    filtered_cloud = random_voxel_filter(init_points, voxel_resolution)
    print('random_voxel_filter use time:', time.time() - start)

    print("Original points num：{num1}, After filtered points num: {num2}"
          .format(num1=init_points.shape[0], num2=len(filtered_cloud)))

    start = time.time()
    downpcd = open3d.geometry.voxel_down_sample(pointcloud, voxel_size=0.01)
    print('voxel_down_sample use time:', time.time() - start)
    open3d.visualization.draw_geometries([downpcd], window_name='Open3D_2')


    # 显示滤波后的点云
    pointcloud.points = open3d.utility.Vector3dVector(filtered_cloud)
    open3d.visualization.draw_geometries([pointcloud]) # window_name='centroid_voxel_filter',height=500,width=600


if __name__ == '__main__':
    main()
