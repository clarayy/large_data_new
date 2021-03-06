#节点状态做为node_labels
#单张图propagation1_6.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import networkx as nx
import os
import random
from itertools import islice
import torch
from scipy.sparse.coo import coo_matrix
import argparse
import math
from random import choice
import community as community_louvain
#变量：B，sn范围,每个sn张数 ，Infectionrate, Roundtime，data文件名

N = 67#记得必须改
fname = "high67"
B = np.load(fname+".npy")#读取固定的图
Gnw = nx.DiGraph(B)
G = nx.read_edgelist('./'+fname+'_weight.txt',nodetype = int,data=(('weight',float),),create_using=nx.DiGraph())
obn = int(N*0.5)  #有一半的节点可以观测到
observenode = random.sample(range(0,N-1),obn)#每次都随机选还是固定呢
#observenode= [i for i in range(62)]
#33;ob=0.5
observenode = [22, 45, 21, 39, 64, 62, 60, 40, 63, 50, 56, 46, 43, 35, 44, 37, 65, 7, 16, 28, 31, 32, 4, 38, 3, 29, 49, 15, 47, 36, 13, 51, 58]
#2068;ob=0.5
#observenode = [2021, 1444, 1406, 1786, 2028, 1336, 227, 1334, 1856, 959, 1885, 437, 519, 1788, 916, 550, 1482, 533, 1838, 1020, 655, 720, 1322, 2003, 2064, 1553, 1798, 634, 973, 512, 472, 481, 1970, 1641, 1030, 21, 445, 386, 1326, 1111, 434, 390, 1115, 1165, 1257, 466, 1519, 1922, 1663, 1332, 286, 32, 1073, 1967, 1013, 187, 1649, 342, 1453, 1791, 389, 1293, 1839, 1727, 1716, 1033, 1749, 438, 330, 75, 1318, 993, 1005, 126, 563, 1554, 974, 1212, 1546, 247, 274, 970, 2046, 2059, 59, 1496, 1944, 1280, 1239, 1998, 34, 488, 1151, 653, 351, 1614, 659, 155, 464, 1026, 320, 347, 494, 1928, 824, 914, 419, 154, 1633, 597, 1586, 1917, 738, 1003, 263, 1643, 1099, 1861, 868, 222, 1462, 1181, 932, 373, 812, 397, 362, 1392, 548, 225, 507, 946, 1413, 1501, 587, 1720, 1424, 1438, 763, 253, 1709, 858, 1167, 459, 567, 500, 893, 13, 1472, 1814, 1373, 1262, 918, 189, 193, 920, 710, 402, 1278, 978, 1112, 1361, 1241, 769, 175, 310, 1029, 1753, 1921, 809, 1722, 377, 1902, 1930, 1863, 440, 76, 1141, 1744, 452, 1937, 1233, 783, 556, 1222, 508, 820, 905, 2012, 958, 841, 848, 1897, 1671, 1098, 1837, 1227, 1634, 1900, 47, 823, 1331, 1933, 2056, 568, 486, 1540, 1420, 1192, 1901, 1324, 761, 728, 301, 1045, 336, 930, 194, 872, 1439, 1122, 1473, 998, 726, 951, 217, 888, 44, 2027, 54, 1910, 1682, 691, 1737, 425, 1497, 211, 1116, 1833, 477, 1766, 457, 50, 1493, 346, 579, 1644, 743, 625, 2002, 161, 1819, 483, 778, 1127, 450, 1043, 1872, 1594, 736, 12, 2029, 1070, 1159, 236, 663, 1877, 1040, 856, 1660, 46, 796, 906, 191, 69, 1559, 1191, 52, 1004, 1074, 551, 1957, 1748, 1981, 467, 987, 240, 1247, 1504, 366, 1875, 1769, 1255, 1144, 1064, 664, 1827, 94, 160, 49, 1796, 1966, 1177, 1443, 2049, 256, 607, 344, 620, 237, 273, 1832, 143, 924, 471, 133, 1154, 414, 1578, 909, 1498, 2065, 1927, 616, 1035, 17, 1781, 1768, 383, 1349, 545, 877, 1794, 1355, 1723, 1603, 148, 1825, 1956, 713, 1091, 747, 1242, 1580, 830, 1404, 1460, 241, 2019, 398, 1149, 1777, 1711, 1077, 1002, 1232, 1369, 915, 721, 756, 1597, 927, 150, 1009, 731, 1628, 470, 1289, 1729, 919, 1449, 1083, 1380, 80, 1218, 1213, 2015, 1582, 199, 814, 1150, 1492, 622, 178, 786, 89, 1525, 698, 1357, 1446, 277, 1607, 157, 704, 1055, 1028, 358, 553, 1778, 27, 860, 1662, 894, 1163, 514, 884, 355, 1110, 341, 1419, 1508, 1477, 1092, 1335, 268, 532, 1265, 559, 643, 1891, 490, 984, 444, 1752, 1281, 288, 1656, 558, 1495, 2034, 319, 2037, 571, 1063, 1445, 1044, 1893, 1418, 1256, 1882, 181, 190, 1148, 886, 1069, 1264, 1201, 1137, 1522, 1743, 1479, 1469, 335, 1829, 289, 1760, 619, 74, 1666, 1637, 1894, 295, 1533, 1428, 1347, 1468, 1869, 1189, 943, 38, 969, 1471, 393, 613, 1736, 269, 233, 1432, 496, 1085, 423, 800, 86, 1627, 748, 1734, 1645, 701, 2006, 1865, 1023, 376, 1696, 1260, 1038, 1836, 3, 1524, 939, 208, 1840, 1342, 836, 430, 2047, 1572, 611, 865, 1648, 1951, 531, 1747, 666, 1667, 712, 614, 118, 1678, 361, 1484, 1435, 784, 1129, 895, 1617, 1683, 1962, 965, 228, 844, 484, 1574, 806, 1605, 1509, 585, 100, 1724, 202, 1270, 282, 697, 1078, 1567, 1205, 329, 637, 705, 967, 1936, 536, 201, 1351, 180, 1892, 1622, 1681, 1670, 293, 1160, 1180, 1565, 1142, 1968, 1701, 179, 1726, 859, 197, 520, 986, 257, 1718, 20, 1989, 527, 37, 751, 186, 1226, 706, 318, 1253, 209, 1691, 4, 308, 648, 843, 122, 498, 934, 1294, 365, 1465, 451, 1434, 996, 62, 1467, 658, 15, 897, 2025, 1047, 1054, 1217, 741, 908, 119, 1821, 405, 93, 11, 110, 1403, 223, 1124, 1389, 432, 1168, 79, 1878, 1475, 1378, 230, 1531, 1325, 475, 1672, 941, 1275, 1871, 1759, 316, 575, 262, 762, 1053, 140, 184, 610, 5, 1046, 530, 29, 285, 249, 1007, 421, 242, 1978, 1503, 1268, 1309, 1860, 1451, 1119, 1909, 379, 1581, 1164, 570, 771, 1984, 1569, 1384, 1657, 977, 732, 1297, 963, 1896, 1101, 1382, 549, 1065, 1757, 1442, 18, 1478, 1834, 1153, 595, 1344, 1579, 1646, 476, 1039, 40, 515, 1631, 81, 662, 1517, 1952, 1808, 287, 505, 583, 1319, 1032, 418, 677, 1298, 221, 113, 1175, 129, 246, 1714, 1550, 1843, 1068, 1130, 2024, 350, 333, 1919, 518, 1859, 656, 581, 730, 540, 759, 1374, 1867, 261, 528, 1126, 792, 315, 1651, 1156, 2001, 104, 1095, 1620, 1204, 1000, 1606, 235, 1292, 136, 1288, 281, 542, 1458, 788, 1061, 1136, 1266, 899, 1207, 441, 294, 1948, 562, 975, 593, 889, 742, 90, 2032, 487, 794, 947, 1584, 921, 566, 1845, 983, 338, 107, 1108, 28, 1765, 360, 1888, 670, 85, 42, 936, 1368, 801, 1721, 1398, 77, 195, 1223, 1433, 1816, 675, 1466, 206, 1010, 1415, 543, 1249, 1328, 1986, 504, 302, 16, 1283, 1703, 53, 1694, 272, 1041, 1178, 654, 1430, 1231, 700, 680, 737, 985, 1947, 0, 276, 1001, 231, 940, 314, 586, 1548, 374, 1991, 718, 1252, 1520, 1850, 455, 808, 1939, 340, 1750, 612, 442, 640, 1536, 2045, 1190, 1801, 1353, 1320, 210, 1480, 873, 863, 1770, 207, 661, 510, 1272, 99, 1899, 328, 152, 892, 172, 169, 1306, 1945, 162, 352, 896, 1166, 106, 1352, 2048, 870, 1437, 1779, 629, 348, 688, 937, 1431, 735, 1379, 1625, 1235, 1173, 19, 645, 764, 626, 631, 1535, 1363, 679, 460, 2010, 1587, 1690, 1959, 1719, 1705, 1767, 283, 1793, 754, 88, 1087, 212, 1296, 734, 727, 926, 580, 1254, 672, 1518, 953, 96, 1542, 1715, 1738, 929, 1693, 461, 1564, 297, 1195, 1291, 1185, 1203, 855, 258, 1626, 829, 111, 815, 1818, 1530, 1802, 109, 506, 427, 98, 1237, 2061, 685, 185, 1880, 1792, 1774, 1858, 410, 1481, 1642, 135, 760, 1024, 1105, 264, 1635, 2050, 1243, 972, 1220, 632, 1062, 1407, 1585, 1601, 803, 739, 1809, 1317, 1461, 1385, 1490, 1624, 1107, 1776, 838, 1555, 1456, 1868, 58, 1284, 1904, 1139, 1330, 604, 1134, 60, 1248, 1898, 750, 903, 1486, 1608, 173, 1547, 1345, 981, 64, 278, 1448, 1599, 1733, 554, 364, 1079, 446, 850, 744, 1376, 1772, 1171, 1598, 714, 535, 1996, 1613, 591, 87, 1290, 322, 306, 369, 1842, 407, 1602, 1817, 1427, 1338, 1790, 910]
#206,ob=0.1
#observenode = [1195, 635, 1052, 1245, 1304, 33, 40, 636, 1797, 763, 1523, 1241, 848, 1310, 751, 593, 388, 1226, 295, 1283, 1340, 1628, 1774, 1532, 398, 1005, 1280, 701, 846, 936, 1974, 285, 424, 42, 1467, 876, 812, 206, 1821, 1524, 825, 505, 1880, 1369, 1733, 1391, 1849, 1074, 1516, 817, 1202, 971, 1551, 480, 1418, 1767, 1805, 1893, 1986, 1636, 952, 200, 2032, 1966, 1781, 628, 956, 1903, 620, 615, 809, 441, 425, 924, 866, 1992, 1358, 1464, 518, 614, 1203, 1382, 1718, 1303, 1794, 1165, 1205, 1631, 642, 115, 1909, 2012, 1837, 697, 1929, 830, 1873, 353, 1409, 538, 1091, 18, 124, 1817, 1894, 1995, 557, 911, 1902, 1860, 1918, 778, 705, 1213, 1421, 600, 318, 1569, 1605, 252, 493, 693, 1674, 831, 1568, 45, 2061, 1413, 1393, 258, 2022, 867, 566, 1384, 1653, 235, 1306, 386, 1037, 1872, 2013, 1415, 879, 378, 726, 1332, 714, 2041, 67, 198, 483, 668, 73, 451, 32, 704, 1755, 1545, 709, 1603, 1225, 2066, 651, 1432, 1350, 150, 720, 1572, 1655, 93, 31, 1219, 219, 1579, 1220, 1728, 80, 983, 2000, 161, 1716, 706, 1877, 382, 156, 1582, 717, 674, 1453, 1776, 1019, 166, 1333, 1086, 71, 1804, 1847, 686, 880, 1748, 144, 1813, 201, 456, 1726, 526]
#print(obn,observenode)
#partition = community_louvain.best_partition(G)
#np.save("BA500_partition.npy",partition)
partition=np.load(fname+'_p'+'.npy',allow_pickle = True).item()
graph_labels = []
graph_labels_class=[]
countedge = []
countadj = []
part=0.1
InfectionRate = 0.5#概率太大，10轮感染1400个节点
Roundtime = 3
def adjConcat(a, b):    #合并对角线矩阵
    lena = len(a)
    lenb = len(b)
    left = np.row_stack((a, np.zeros((lenb, lena),dtype=int)))  # 先将a和一个len(b)*len(a)的零矩阵垂直拼接，得到左半边
    right = np.row_stack((np.zeros((lena, lenb),dtype=int), b))  # 再将一个len(a)*len(b)的零矩阵和b垂直拼接，得到右半边
    result = np.hstack((left, right))  # 将左右矩阵水平拼接
    return result
def genGraph(sn,datadir,bmname):

#感染过程
    node = list(map(int,G.nodes))#图中节点列表，元素转化为整数型
    S = node
    I = []

    j=0
    while j<1:
        #start_node = random.choice(node)#1个初始感染节点
        start_node = sn#1个初始感染节点
        I.append(start_node)
        S.remove(start_node)
        j=j+1
    
    print(start_node)
    
    new_G_small = nx.DiGraph()
    ob_G = nx.DiGraph()
    count = [1]
    statechange = []
    edgechange = []
    edgeweight = []
    weight_s = 1
    #for i in range(Roundtime):       #####从I开始遍历，小于边的概率就感染
    while len(I)<=part*len(G.nodes()):
        for nbr, datadict in G.adj.items():#遍历G的所有节点，nbr节点名称，datadict与节点相连的边
            if int(nbr) in I:            
                for key in datadict:
                    if int(key) in S:
                        rate = G.get_edge_data(int(nbr),int(key))['weight']   #I->S 
                        if random.random() <= rate: 
                            statechange.append(int(key))
                            new_G_small.add_edge(int(nbr),int(key))     #I->S
        statechange=list(set(statechange))
        for i in statechange:
            S.remove(i)
            I.append(i)
        if len(I)==1:
            break
        count.append(len(I))
        statechange = []
    ob_G = G.subgraph(observenode)
    relabel_ob_G = nx.DiGraph()
    mapping={}
    relabel_ob_G_x={}
    it=0
    for ob_n in ob_G.nodes():
        mapping[ob_n]=it
        if ob_n in I:
            relabel_ob_G_x[it]=2     #I 状态
        else:
            relabel_ob_G_x[it]=1      #S状态
        it+=1
    relabel_ob_G=nx.relabel_nodes(ob_G, mapping)     #此时两种图的节点命名应该是不同的

    g_n_i=len(relabel_ob_G_x)
    for g_n in observenode:
        if g_n not in ob_G.nodes():
            if g_n in I:
                relabel_ob_G_x[g_n_i]=2
            else:
                relabel_ob_G_x[g_n_i]=1
            g_n_i +=1
    #print("relabel_ob_G_x:",relabel_ob_G_x)
    
    perfix = os.path.join(datadir,bmname)
    filename_node_labels = perfix + '_node_labels.txt'
    # filename_center = perfix + '_jordancenter.txt'
    # filename_center1 = perfix + '_jordancenter1.txt'
    # filename_unbet = perfix + '_unbet.txt'
    # filename_discen = perfix + '_discen.txt'
    # filename_dynage = perfix + '_dynage.txt'
    #new_G_small表示感染图
    #new_G表示感染图加上未感染的节点，
    #将new_G中的邻接矩阵扩展到G
    # new_G = nx.DiGraph()
    # new_G.add_nodes_from(i for i in range(N))
    # new_G.add_nodes_from(new_G_small.nodes())    #不增加单独节点看实验效果如何，max_nodes=100时，max graph size 是否为100
    # new_G.add_edges_from(new_G_small.edges())
    # #Jordan_center  = nx.center(new_G)#非全连接图不能计算

    if not nx.is_empty(new_G_small):
        adj_matrix_relabel = nx.adjacency_matrix(relabel_ob_G).todense()   #为了获得adj与节点状态相对应的A和X
        island = len(observenode)-len(ob_G.nodes())
        adj_matrix = adjConcat(adj_matrix_relabel,np.zeros((island,island),dtype=int))
        #Jordan_center  = nx.center(new_G_small)

        # #dynamic ages
        # frozen_graph = nx.freeze(new_G_small)#G被冷冻为frozen_graph，不会改变
        # unfrozen_graph = nx.Graph(frozen_graph)#删除节点在非冷冻图上进行，冷冻图不变
        # AS = nx.adjacency_spectrum(frozen_graph)#邻接矩阵特征值
        # m = np.real(AS).round(4).max()
        # all_nodes = new_G_small.nodes
        # #print(all_nodes)
        # da = {}                             ###!!!!!字典才对
        # for i in all_nodes:
        #     unfrozen_graph.remove_node(i)
        #     AS1 = nx.adjacency_spectrum(unfrozen_graph)
        #     m1 = np.real(AS1).round(4).max()
        #     da[i] = float(format(abs(m-m1)/m,'.4f'))   #单独运算看对不对
        #     unfrozen_graph = nx.Graph(frozen_graph)
        # dynage = max(da, key=lambda x: da[x]) 
        # # dynage=1
        # with open(filename_dynage,'a') as dynagef:
        #     dynagef.write(str(dynage))
        #     dynagef.write('\n')

        # nx.draw(new_G_small,with_labels = True)
        # plt.show()
    #np.savetxt('datatest.txt',adj_matrix)
    #print(adj_matrix.shape)
        countadj_now=adj_matrix.shape[1]
        countadj.append(adj_matrix.shape[1])
        countedge.append(ob_G.number_of_edges())
        # with open(filename_center,'a') as centerf:
        #     centerf.write(str(choice(Jordan_center)))    #Jordan center随机选，按理来说差别可能不大，实际有差别
        #     centerf.write('\n')
        # with open(filename_center1,'a') as centerf1:
        #     centerf1.write(str(Jordan_center[0]))    #Jordan center选第一个
        #     centerf1.write('\n')
        with open(filename_node_labels,'a') as labelf:#节点ID作为标签
            for k,v in relabel_ob_G_x.items():
                labelf.write(str(v))
                labelf.write('\n')
        graph_labels.append(start_node)
        graph_labels_class.append(partition[start_node])#所属的类
    else:
        adj_matrix=[]
        countadj_now=[]


    return (adj_matrix,countadj_now)

#可以自己造邻接矩阵，行和列的范围从adj_matrix.shape开始增加
#直接生成data_A.txt  边的邻接矩阵
def data_A(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_A = perfix + '_A.txt'
    filename_node_labels = perfix + '_dre_node_labels.txt'
    sum_ca_now = 0
    rand_graphs = [30,40,50]
    graphs=10
    nodehead=30
    nodetail=67
    class0= [10, 18, 24, 26, 36, 41, 46, 59, 61, 63, 88, 97, 115, 120, 121, 129, 130, 131, 153, 157, 162, 164, 178, 183, 187, 195, 210, 211, 212, 214, 220, 225, 236, 258, 266, 271, 281, 284, 293, 303, 304, 325, 327, 332, 334, 336, 359, 366, 370, 380, 384, 393, 394, 399, 413, 428, 430, 432, 450, 454, 457, 474, 492, 499, 518, 533, 546, 558, 565, 570, 572, 576, 579, 583, 614, 616, 619, 635, 636, 639, 643, 650, 651, 657, 658, 659, 665, 667, 672, 673, 682, 710, 716, 719, 728, 739, 740, 744, 758, 762, 764, 773, 785, 787, 788, 795, 797, 812, 826, 831, 835, 836, 850, 852, 854, 860, 872, 879, 882, 890, 902, 903, 904, 906, 923, 947, 954, 958, 971, 981, 982, 989, 1014, 1016, 1022, 1036, 1040, 1042, 1051, 1057, 1064, 1068, 1072, 1083, 1085, 1094, 1096, 1108, 1111, 1120, 1141, 1144, 1162, 1173, 1186, 1189, 1190, 1210, 1217, 1234, 1236, 1240, 1258, 1269, 1284, 1285, 1288, 1292, 1298, 1304, 1317, 1325, 1333, 1336, 1340, 1343, 1353, 1355, 1358, 1369, 1400, 1402, 1405, 1418, 1420, 1422, 1427, 1430, 1431, 1453, 1458, 1459, 1461, 1464, 1481, 1482, 1486, 1499, 1501, 1512, 1521, 1522, 1525, 1532, 1541, 1546, 1581, 1583, 1592, 1607, 1613, 1616, 1629, 1631, 1649, 1652, 1653, 1659, 1667, 1668, 1681, 1705, 1706, 1710, 1715, 1718, 1736, 1743, 1754, 1765, 1766, 1781, 1786, 1808, 1812, 1816, 1818, 1819, 1825, 1832, 1833, 1847, 1852, 1861, 1865, 1866, 1872, 1882, 1892, 1913, 1917, 1920, 1923, 1927, 1928, 1929, 1945, 1959, 1979, 1980, 2007, 2008, 2011, 2041, 2046, 2047, 2058, 2062, 2067]
    class1= [1, 2, 6, 7, 8, 15, 17, 21, 25, 29, 33, 34, 35, 38, 43, 45, 47, 50, 51, 54, 55, 58, 65, 66, 71, 74, 79, 83, 91, 94, 99, 100, 102, 104, 106, 108, 110, 112, 113, 114, 116, 126, 128, 132, 136, 137, 145, 149, 155, 156, 159, 163, 168, 169, 172, 175, 186, 190, 197, 198, 204, 206, 208, 213, 215, 219, 222, 229, 233, 235, 237, 238, 239, 241, 247, 262, 263, 273, 276, 278, 291, 292, 295, 298, 309, 312, 315, 321, 322, 326, 335, 342, 345, 346, 349, 363, 369, 372, 374, 375, 377, 381, 382, 387, 392, 397, 403, 408, 409, 410, 414, 427, 433, 435, 436, 441, 444, 445, 447, 448, 465, 466, 467, 471, 485, 487, 488, 490, 493, 496, 500, 505, 511, 513, 516, 519, 526, 527, 531, 536, 547, 553, 554, 555, 556, 563, 568, 574, 586, 590, 591, 592, 594, 597, 608, 610, 612, 613, 623, 628, 631, 652, 654, 662, 664, 669, 671, 674, 676, 678, 679, 680, 685, 689, 690, 691, 694, 698, 699, 701, 705, 706, 708, 712, 718, 720, 721, 722, 727, 729, 736, 745, 746, 747, 750, 752, 753, 760, 761, 766, 769, 770, 772, 775, 777, 779, 780, 786, 791, 800, 804, 806, 807, 814, 819, 820, 822, 824, 825, 837, 839, 841, 844, 845, 846, 847, 851, 856, 863, 870, 871, 873, 874, 880, 881, 891, 898, 899, 900, 905, 912, 914, 915, 920, 921, 925, 930, 932, 935, 938, 939, 942, 949, 957, 960, 965, 967, 968, 973, 974, 975, 980, 986, 987, 995, 996, 997, 1000, 1002, 1010, 1013, 1015, 1017, 1024, 1026, 1027, 1030, 1056, 1062, 1063, 1075, 1076, 1081, 1101, 1107, 1110, 1112, 1113, 1114, 1117, 1119, 1122, 1124, 1126, 1136, 1140, 1143, 1146, 1147, 1148, 1158, 1160, 1161, 1164, 1169, 1170, 1171, 1175, 1176, 1178, 1183, 1194, 1195, 1196, 1197, 1198, 1200, 1207, 1211, 1212, 1213, 1216, 1218, 1225, 1228, 1230, 1231, 1238, 1242, 1245, 1247, 1248, 1253, 1259, 1261, 1272, 1274, 1277, 1280, 1287, 1291, 1293, 1294, 1302, 1303, 1307, 1308, 1311, 1323, 1327, 1329, 1335, 1342, 1347, 1351, 1352, 1356, 1357, 1362, 1363, 1366, 1367, 1374, 1375, 1377, 1378, 1381, 1383, 1384, 1393, 1401, 1410, 1414, 1416, 1417, 1433, 1436, 1438, 1442, 1444, 1446, 1467, 1470, 1471, 1473, 1478, 1484, 1485, 1487, 1489, 1491, 1492, 1497, 1500, 1504, 1505, 1506, 1513, 1515, 1516, 1518, 1530, 1535, 1538, 1539, 1542, 1544, 1548, 1552, 1556, 1558, 1561, 1562, 1569, 1570, 1574, 1575, 1579, 1582, 1586, 1588, 1598, 1602, 1603, 1605, 1608, 1610, 1611, 1622, 1624, 1634, 1635, 1638, 1641, 1642, 1644, 1657, 1660, 1663, 1664, 1671, 1672, 1673, 1675, 1684, 1685, 1691, 1692, 1694, 1697, 1698, 1701, 1711, 1714, 1716, 1719, 1723, 1724, 1728, 1731, 1733, 1738, 1739, 1742, 1744, 1749, 1752, 1767, 1771, 1774, 1787, 1789, 1795, 1796, 1797, 1806, 1810, 1813, 1815, 1823, 1824, 1834, 1841, 1842, 1846, 1848, 1849, 1850, 1851, 1856, 1863, 1868, 1881, 1884, 1885, 1886, 1897, 1901, 1902, 1906, 1907, 1912, 1914, 1915, 1924, 1932, 1936, 1939, 1940, 1944, 1949, 1953, 1954, 1967, 1987, 1998, 2021, 2024, 2029, 2030, 2034, 2035, 2036, 2038, 2039, 2042, 2045, 2048, 2049, 2051, 2055, 2063]
    class2= [0, 4, 5, 19, 42, 60, 62, 64, 69, 73, 80, 85, 89, 111, 118, 119, 122, 124, 140, 141, 143, 148, 150, 161, 170, 171, 174, 176, 181, 182, 191, 196, 202, 203, 205, 231, 232, 243, 260, 265, 270, 279, 280, 285, 287, 290, 297, 302, 305, 317, 318, 319, 343, 348, 357, 360, 373, 376, 378, 386, 388, 391, 398, 406, 407, 411, 416, 418, 422, 431, 438, 452, 460, 463, 468, 472, 478, 480, 482, 486, 489, 491, 503, 504, 508, 510, 514, 523, 534, 541, 542, 545, 552, 559, 573, 577, 585, 587, 588, 589, 598, 601, 602, 604, 606, 609, 626, 627, 633, 640, 641, 642, 647, 648, 670, 677, 693, 709, 713, 715, 725, 726, 730, 732, 735, 737, 742, 765, 767, 789, 796, 803, 809, 811, 816, 828, 829, 832, 833, 840, 842, 848, 849, 855, 864, 866, 868, 869, 878, 883, 889, 893, 897, 901, 909, 910, 911, 917, 928, 934, 936, 937, 941, 943, 944, 956, 959, 962, 969, 972, 977, 984, 990, 992, 993, 1005, 1006, 1008, 1011, 1019, 1021, 1023, 1033, 1041, 1049, 1053, 1059, 1060, 1065, 1071, 1074, 1084, 1086, 1090, 1095, 1098, 1100, 1115, 1123, 1128, 1132, 1133, 1134, 1142, 1157, 1159, 1172, 1182, 1184, 1188, 1192, 1193, 1209, 1214, 1219, 1221, 1223, 1227, 1232, 1239, 1243, 1249, 1250, 1252, 1254, 1255, 1257, 1262, 1263, 1265, 1276, 1283, 1290, 1295, 1309, 1312, 1313, 1314, 1315, 1326, 1328, 1332, 1334, 1338, 1345, 1346, 1365, 1370, 1380, 1391, 1396, 1398, 1403, 1409, 1415, 1419, 1425, 1428, 1434, 1435, 1437, 1441, 1443, 1449, 1460, 1468, 1475, 1479, 1483, 1494, 1502, 1503, 1507, 1509, 1511, 1514, 1529, 1533, 1547, 1550, 1555, 1557, 1559, 1567, 1568, 1573, 1578, 1580, 1584, 1594, 1600, 1601, 1604, 1617, 1623, 1637, 1639, 1645, 1647, 1651, 1674, 1689, 1690, 1703, 1704, 1712, 1717, 1722, 1726, 1734, 1741, 1750, 1760, 1761, 1762, 1768, 1772, 1775, 1777, 1783, 1785, 1790, 1792, 1799, 1801, 1802, 1803, 1811, 1817, 1820, 1822, 1829, 1836, 1843, 1870, 1871, 1874, 1878, 1883, 1894, 1895, 1896, 1930, 1948, 1951, 1957, 1958, 1966, 1970, 1971, 1972, 1978, 1982, 1986, 1988, 1990, 1992, 2001, 2005, 2014, 2015, 2018, 2019, 2022, 2032, 2037, 2040, 2059, 2066]
    class3= [13, 20, 22, 23, 27, 40, 49, 57, 72, 84, 86, 101, 123, 125, 127, 134, 142, 147, 151, 152, 167, 173, 179, 180, 184, 199, 200, 224, 245, 255, 259, 261, 264, 274, 275, 277, 286, 288, 294, 301, 308, 310, 311, 314, 316, 329, 330, 331, 347, 354, 355, 358, 362, 379, 383, 423, 426, 449, 455, 462, 473, 476, 481, 506, 517, 539, 543, 548, 549, 567, 569, 571, 578, 584, 593, 599, 600, 617, 618, 622, 632, 649, 655, 663, 666, 668, 681, 688, 697, 703, 717, 731, 748, 749, 751, 755, 757, 771, 778, 783, 790, 798, 802, 808, 813, 817, 818, 838, 894, 907, 913, 916, 918, 922, 933, 945, 955, 963, 964, 966, 991, 998, 1009, 1012, 1018, 1020, 1031, 1032, 1035, 1038, 1046, 1055, 1067, 1077, 1078, 1079, 1091, 1106, 1109, 1116, 1131, 1139, 1150, 1152, 1163, 1166, 1181, 1185, 1201, 1237, 1241, 1246, 1267, 1268, 1270, 1271, 1278, 1279, 1281, 1289, 1296, 1319, 1322, 1337, 1348, 1349, 1350, 1361, 1379, 1385, 1387, 1450, 1451, 1454, 1457, 1466, 1469, 1488, 1510, 1520, 1526, 1528, 1534, 1536, 1545, 1563, 1564, 1566, 1577, 1596, 1615, 1625, 1627, 1633, 1655, 1666, 1669, 1677, 1688, 1709, 1732, 1737, 1740, 1746, 1755, 1757, 1764, 1778, 1782, 1784, 1788, 1793, 1800, 1804, 1821, 1826, 1830, 1838, 1839, 1854, 1857, 1869, 1875, 1877, 1879, 1888, 1904, 1909, 1918, 1937, 1942, 1943, 1946, 1956, 1962, 1965, 1974, 1975, 1985, 1991, 1996, 1999, 2000, 2002, 2003, 2010, 2017, 2023, 2025, 2028, 2043, 2060, 2064]
    class4= [3, 9, 11, 12, 14, 16, 28, 30, 31, 32, 37, 39, 44, 48, 52, 53, 56, 67, 68, 70, 75, 76, 77, 78, 81, 82, 87, 90, 92, 93, 95, 96, 98, 103, 105, 107, 109, 117, 133, 135, 138, 139, 144, 146, 154, 158, 160, 165, 166, 177, 185, 188, 189, 192, 193, 194, 201, 207, 209, 216, 217, 218, 221, 223, 226, 227, 228, 230, 234, 240, 242, 244, 246, 248, 249, 250, 251, 252, 253, 254, 256, 257, 267, 268, 269, 272, 282, 283, 289, 296, 299, 300, 306, 307, 313, 320, 323, 324, 328, 333, 337, 338, 339, 340, 341, 344, 350, 351, 352, 353, 356, 361, 364, 365, 367, 368, 371, 385, 389, 390, 395, 396, 400, 401, 402, 404, 405, 412, 415, 417, 419, 420, 421, 424, 425, 429, 434, 437, 439, 440, 442, 443, 446, 451, 453, 456, 458, 459, 461, 464, 469, 470, 475, 477, 479, 483, 484, 494, 495, 497, 498, 501, 502, 507, 509, 512, 515, 520, 521, 522, 524, 525, 528, 529, 530, 532, 535, 537, 538, 540, 544, 550, 551, 557, 560, 561, 562, 564, 566, 575, 580, 581, 582, 595, 596, 603, 605, 607, 611, 615, 620, 621, 624, 625, 629, 630, 634, 637, 638, 644, 645, 646, 653, 656, 660, 661, 675, 683, 684, 686, 687, 692, 695, 696, 700, 702, 704, 707, 711, 714, 723, 724, 733, 734, 738, 741, 743, 754, 756, 759, 763, 768, 774, 776, 781, 782, 784, 792, 793, 794, 799, 801, 805, 810, 815, 821, 823, 827, 830, 834, 843, 853, 857, 858, 859, 861, 862, 865, 867, 875, 876, 877, 884, 885, 886, 887, 888, 892, 895, 896, 908, 919, 924, 926, 927, 929, 931, 940, 946, 948, 950, 951, 952, 953, 961, 970, 976, 978, 979, 983, 985, 988, 994, 999, 1001, 1003, 1004, 1007, 1025, 1028, 1029, 1034, 1037, 1039, 1043, 1044, 1045, 1047, 1048, 1050, 1052, 1054, 1058, 1061, 1066, 1069, 1070, 1073, 1080, 1082, 1087, 1088, 1089, 1092, 1093, 1097, 1099, 1102, 1103, 1104, 1105, 1118, 1121, 1125, 1127, 1129, 1130, 1135, 1137, 1138, 1145, 1149, 1151, 1153, 1154, 1155, 1156, 1165, 1167, 1168, 1174, 1177, 1179, 1180, 1187, 1191, 1199, 1202, 1203, 1204, 1205, 1206, 1208, 1215, 1220, 1222, 1224, 1226, 1229, 1233, 1235, 1244, 1251, 1256, 1260, 1264, 1266, 1273, 1275, 1282, 1286, 1297, 1299, 1300, 1301, 1305, 1306, 1310, 1316, 1318, 1320, 1321, 1324, 1330, 1331, 1339, 1341, 1344, 1354, 1359, 1360, 1364, 1368, 1371, 1372, 1373, 1376, 1382, 1386, 1388, 1389, 1390, 1392, 1394, 1395, 1397, 1399, 1404, 1406, 1407, 1408, 1411, 1412, 1413, 1421, 1423, 1424, 1426, 1429, 1432, 1439, 1440, 1445, 1447, 1448, 1452, 1455, 1456, 1462, 1463, 1465, 1472, 1474, 1476, 1477, 1480, 1490, 1493, 1495, 1496, 1498, 1508, 1517, 1519, 1523, 1524, 1527, 1531, 1537, 1540, 1543, 1549, 1551, 1553, 1554, 1560, 1565, 1571, 1572, 1576, 1585, 1587, 1589, 1590, 1591, 1593, 1595, 1597, 1599, 1606, 1609, 1612, 1614, 1618, 1619, 1620, 1621, 1626, 1628, 1630, 1632, 1636, 1640, 1643, 1646, 1648, 1650, 1654, 1656, 1658, 1661, 1662, 1665, 1670, 1676, 1678, 1679, 1680, 1682, 1683, 1686, 1687, 1693, 1695, 1696, 1699, 1700, 1702, 1707, 1708, 1713, 1720, 1721, 1725, 1727, 1729, 1730, 1735, 1745, 1747, 1748, 1751, 1753, 1756, 1758, 1759, 1763, 1769, 1770, 1773, 1776, 1779, 1780, 1791, 1794, 1798, 1805, 1807, 1809, 1814, 1827, 1828, 1831, 1835, 1837, 1840, 1844, 1845, 1853, 1855, 1858, 1859, 1860, 1862, 1864, 1867, 1873, 1876, 1880, 1887, 1889, 1890, 1891, 1893, 1898, 1899, 1900, 1903, 1905, 1908, 1910, 1911, 1916, 1919, 1921, 1922, 1925, 1926, 1931, 1933, 1934, 1935, 1938, 1941, 1947, 1950, 1952, 1955, 1960, 1961, 1963, 1964, 1968, 1969, 1973, 1976, 1977, 1981, 1983, 1984, 1989, 1993, 1994, 1995, 1997, 2004, 2006, 2009, 2012, 2013, 2016, 2020, 2026, 2027, 2031, 2033, 2044, 2050, 2052, 2053, 2054, 2056, 2057, 2061, 2065]
    #nodelist = class0
    with open(filename_A,'w') as f:
        with open(filename_node_labels,'w') as f1: #节点度记录
            #for nodesn in nodelist:
            for nodesn in range(nodehead,nodetail):
                #for j in range(choice(rand_graphs)):
                for j in range(graphs):
                    adj,ca_now=genGraph(nodesn,datadir,bmname)
                    if len(adj):                      #图为非空，才进行下一步
                        coo_A=coo_matrix(adj)   #邻接矩阵的边的行/列的坐标
                        edge_index = [coo_A.row+1,coo_A.col+1]
                        #node_labels(adj)
                        a=np.array(adj)
                        a=np.sum(a,axis=1)
                        a=a.tolist()
                        for i in range(len(a)):
                            f1.write(str(a[i]))
                            f1.write('\n')
                        if len(countadj)==1:
                            for i in range(len(edge_index[1])):
                                f.write(str(edge_index[0][i])+','+str(edge_index[1][i]))
                                f.write('\n')
                                #print(str(coo_A.row[i])+','+str(coo_A.col[i]))
                        else:
                            for i in range(len(edge_index[1])):
                                f.write(str(edge_index[0][i]+sum_ca_now)+','+str(edge_index[1][i]+sum_ca_now))
                                f.write('\n')
                                #print(str(coo_A.row[i]+sum_ca_now)+','+str(coo_A.col[i]+sum_ca_now))
                        sum_ca_now=sum_ca_now+ca_now
    #/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data
    filename_readme = perfix + 'readme.txt'
    with open(filename_readme,'a') as f:
        #f.write('InfectionRate='+str(InfectionRate)+"\n")
        #f.write('Roundtime='+str(Roundtime)+"\n")
        f.write('[a,b]='+str(nodehead)+','+str(nodetail)+"\n")
        #f.write('nodelist='+str(nodelist)+'\n')
        #f.write('every node graphs='+str(graphs)+"\n")


def main():

    bmname = fname+'_p0.1_ob0.5_m10_2'
    #path = os.path.join('/home/zhang/Documents/pytorch/learn/GraphKernel/rexying_diffpool/diffpool-master/data',bmname)
    path = os.path.join('/home/iot/zcy/usb/copy/myGCN/data/'+fname+'/'+bmname+'/raw')
    
    #path = os.path.join('data',bmname)#调试时生成的文件夹
    if not os.path.exists(path):
        os.makedirs(path)
    perfix = os.path.join(path,bmname)
    filename_readme = perfix+'readme.txt'
    with open(filename_readme,'w') as f:
        f.write('bmname = '+str(bmname)+"\n")
        f.write('N='+str(N)+"\n")
        f.write('底图='+fname+".npy"+"\n")
        f.write(fname+'底图，感染节点占part比例时停止传播，测试集，包括多种对比方法，分5类'+"\n")
        f.write('part='+str(part)+"\n")
        #f.write('val_datatest'+"\n")

    data=open(filename_readme,'a')
    data_A(path,bmname)
    graph_label(path,bmname)
    graph_indicator(path,bmname)
    graph_label_classfication(path,bmname)

    dis_s=0
    for i in countedge:
        dis_s=dis_s+i
    print('sum of edges:',file=data)
    print(str(dis_s)+'\n',file=data)
    s1=0
    for i in countadj:
        s1=s1+i
    print('sum of adj:',file=data)
    print(str(s1)+'\n',file=data)

def graph_indicator(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_indic = perfix + '_graph_indicator.txt'
    with open(filename_graph_indic,'w') as f:
        i=1
        for val in countadj:
            for j in range(int(val)):
                f.write(str(i))
                f.write('\n')
            i=i+1
def graph_label(datadir,bmname):
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels = perfix+ '_graph_labels.txt'
    with open(filename_graph_labels,'w') as f:
        for i in graph_labels:
            f.write(str(i))
            f.write('\n')
def graph_label_classfication(datadir,bmname): #分类后的节点标签
    perfix = os.path.join(datadir,bmname)
    filename_graph_labels_class = perfix+ '_graph_labels_class.txt'
    with open(filename_graph_labels_class,'w') as f:
        for i in graph_labels_class:
            f.write(str(i))
            f.write('\n')

if __name__ == "__main__":
    main()
