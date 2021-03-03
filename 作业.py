# -*- coding:utf-8 -*-
from tkinter import scrolledtext

import pandas as pd
import os
import tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import seaborn as sns
import numpy as np
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller as ADF
from tkinter import *


# 函数们
def mainwindows():  # 创建主窗口
    global root
    root = tkinter.Tk()
    root.title("游戏数据分析")
    root.geometry("420x420")
    root["background"] = '#F8F8FF'
    col_count, row_count = root.grid_size()

    for col in range(col_count):
        root.grid_columnconfigure(col, minsize=20)

    for row in range(row_count):
        root.grid_rowconfigure(row, minsize=20)

    def quitmain():
        root.quit()  # 结束主循环
        root.destroy()  # 销毁窗口

    # 创建按钮
    button0 = tkinter.Button(root, text="数据总览", command=datascreening, width=15, height=5, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button1 = tkinter.Button(root, text="各地区销量分析", command=area_analysis, width=15, height=5, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button2 = tkinter.Button(root, text="各区域销量趋势", command=area_sales, width=15, height=5, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button3 = tkinter.Button(root, text="筛选", command=choose, width=15, height=5, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button4 = tkinter.Button(root, text="搜索", command=search, width=15, height=5, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button5 = tkinter.Button(root, text="全球销量时序预测", command=Timeseriesprediction, width=15, height=5, bg='#F0FFFF',
                             fg='#191970', activeforeground='#F0F8FF', activebackground='#4682B4')
    buttonX = tkinter.Button(root, text="退出", command=quitmain, width=15, height=5, bg='#F0FFFF',
                             activeforeground='#F0F8FF', activebackground='#4862B4')
    # 放置按钮
    button0.grid(row=0, column=1, padx=10, pady=3)
    button1.grid(row=0, column=2, padx=10, pady=3)
    button2.grid(row=1, column=1, padx=10, pady=3)
    button3.grid(row=1, column=2, padx=10, pady=3)
    button4.grid(row=2, column=1, padx=10, pady=3)
    button5.grid(row=2, column=2, padx=10, pady=3)
    buttonX.grid(row=3, column=3, padx=10, pady=3)

    root.mainloop()


def datascreening():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("平台销量分布及游戏种类销量分布")  # 窗口标题
    root1.geometry("1080x1080")
    root1["background"] = '#F8F8FF'
    # 填写函数区
    #出版社销量分布情况
    Pub_Global_Sales = df.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False)[:20]
    # 平台的销量分布情况
    Plat_Global_Sales = df.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False)

    # 游戏类型销量分布情况
    Genre_Global_Sales = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)

    # 可视化
    f, ax = plt.subplots(1, 3, figsize=(25, 8), dpi=100)
    sns.barplot(Pub_Global_Sales.values, Pub_Global_Sales.index, ax=ax[0])
    ax[0].set_title('Publisher_Global_Sales')
    sns.barplot(Plat_Global_Sales.values, Plat_Global_Sales.index, ax=ax[1])
    ax[1].set_title('Platform_Global_Sales')
    sns.barplot(Genre_Global_Sales.values, Genre_Global_Sales.index, ax=ax[2])
    ax[2].set_title('Genre_Global_Sales')

    # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
    canvas = FigureCanvasTkAgg(f, master=root1)
    canvas.draw()  # 注意show方法已经过时了,这里改用draw
    canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                fill=tkinter.BOTH,  # 填充方式
                                expand=tkinter.YES)  # 随窗口大小调整而调整

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    button0 = tkinter.Button(root1, text="退出", command=quit, width=15, height=3, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0.pack(side=tkinter.BOTTOM)
    root1.mainloop()
    mainwindows()


def area_sales():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("各区域销量趋势")
    root1.geometry("720x720")
    root1["background"] = '#F8F8FF'
    # 填写函数区
    M = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    df5market_p = pd.pivot_table(df, index='Year', values=M, aggfunc=np.sum)
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(data=df5market_p)
    plt.title('Trends of different areas')

    # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
    canvas = FigureCanvasTkAgg(fig, master=root1)
    canvas.draw()  # 注意show方法已经过时了,这里改用draw
    canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                fill=tkinter.BOTH,  # 填充方式
                                expand=tkinter.YES)  # 随窗口大小调整而调整

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    button0 = tkinter.Button(root1, text="退出", command=quit, width=15, height=3, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0.pack(side=tkinter.BOTTOM)
    root1.mainloop()
    mainwindows()


def area_analysis():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("不同地区受欢迎平台分布")
    root1.geometry("1080x1080")
    root1["background"] = '#F8F8FF'
    # 填写函数区
    area = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 6), sharex=True, sharey=True)
    fig.tight_layout()
    for i in range(4):
        df.pivot_table(index='Platform', values=area[i], aggfunc='sum').sort_values(area[i], ascending=False).plot.bar(
            ax=ax.ravel()[i])
    canvas = FigureCanvasTkAgg(fig, master=root1)
    canvas.draw()  # 注意show方法已经过时了,这里改用draw
    canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                fill=tkinter.BOTH,  # 填充方式
                                expand=tkinter.YES)  # 随窗口大小调整而调整

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    button0 = tkinter.Button(root1, text="退出", command=quit, width=15, height=3, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0.pack(side=tkinter.BOTTOM)
    root1.mainloop()
    mainwindows()


def Timeseriesprediction():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("全球销量时序预测")
    root1.geometry("720x720")
    root1["background"] = '#F8F8FF'
    # 填写函数区
    fig = plt.figure(figsize=(10, 6))  # 确定图像大小

    new_data = pd.pivot_table(df, index='Year', values='Global_Sales', aggfunc='sum')
    # new_data=pd.DataFrame(df.groupby('Year').agg({'Global_Sales':np.sum}))
    new_data_1 = new_data.diff()  # 为使p值减小，进行一阶差分操作
    # 清除inf和nan值
    new_data_1[np.isnan(new_data_1)] = 0
    new_data_1[np.isinf(new_data_1)] = 0
    result = ADF(new_data_1)  # 计算p值
    print(result)
    # sns.lineplot(data=new_data)
    # p值为0.0015702136500286477，可以极显著的拒绝原假设，认为数据平稳。

    # 由ACF、PACF图，进行初步定价，均为拖尾性，可建立ARMA(1,1)

    model = ARMA(new_data_1, order=(1, 1))  # 建立模型
    predict_ts = model.fit().predict()  # 取得结果

    # predict_ts = predict_ts.shift()
    # 预测结果对比可视化
    plt.plot(new_data, label='Origin_diff')  # 使用差分前的数据
    plt.plot(predict_ts, label='Predict_diff')
    plt.legend(loc='best')

    # 效果其实不好。。。

    # 将绘制的图形显示到tkinter:创建属于root的canvas画布,并将图f置于画布上
    canvas = FigureCanvasTkAgg(fig, master=root1)
    canvas.draw()  # 注意show方法已经过时了,这里改用draw
    canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                fill=tkinter.BOTH,  # 填充方式
                                expand=tkinter.YES)  # 随窗口大小调整而调整

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    button0 = tkinter.Button(root1, text="退出", command=quit, width=15, height=3, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0.pack(side=tkinter.BOTTOM)
    root1.mainloop()
    mainwindows()


def choose():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("筛选")
    root1.geometry("1080x720")
    root1["background"] = '#F8F8FF'
    frame1 = Frame(root1)
    frame2 = Frame(root1)
    frame1["background"] = '#F8F8FF'
    frame2["background"] = '#F8F8FF'
    entry1 = tkinter.Entry(frame2)
    entry2 = tkinter.Entry(frame2)
    text1 = scrolledtext.ScrolledText(frame1, width=142, height=40)

    # 填写函数区
    def sortYear():
        pd.set_option('display.max_rows', None)  # 设置最大列数显示，解决打印不完整，显示省略号问题
        string1 = entry1.get()  # 获取数据
        inn1 = int(string1)  # 转换为int
        string2 = entry2.get()
        inn2 = int(string2)
        df2 = df[['Name', 'Year', 'Platform', 'Genre', 'Publisher', 'Global_Sales']][
            (df["Year"] >= inn1) & (df["Year"] <= inn2)]  # 筛选
        text1.delete('1.0', 'end')  # 清空text
        text1.insert(tkinter.constants.END, chars=str(df2))  # 打印筛选结果

    def sortSell():
        pd.set_option('display.max_rows', None)
        string1 = entry1.get()
        inn1 = float(string1)
        string2 = entry2.get()
        inn2 = float(string2)
        df2 = df[['Name', 'Year', 'Platform', 'Genre', 'Publisher', 'Global_Sales']][
            (df["Global_Sales"] >= inn1) & (df["Global_Sales"] <= inn2)]
        text1.delete('1.0', 'end')
        text1.insert(tkinter.constants.END, chars=str(df2))

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    label1 = tkinter.Label(frame2, text="最小值", width=5, height=1, bg='#F8F8FF')
    label2 = tkinter.Label(frame2, text="最大值", width=5, height=1, bg='#F8F8FF')
    button1 = tkinter.Button(frame2, text="按年份筛选", command=sortYear, width=10, height=1, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button2 = tkinter.Button(frame2, text="按销量筛选", command=sortSell, width=10, height=1, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0 = tkinter.Button(frame2, text="退出", command=quit, width=10, height=1, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    frame1.grid(row=2, column=1, padx=20, pady=3)
    frame2.grid(row=1, column=1, padx=20, pady=3)
    entry1.grid(row=1, column=1, padx=20, pady=3)
    entry2.grid(row=2, column=1, padx=20, pady=3)
    text1.grid(row=10, column=3, padx=20, pady=3)
    button0.grid(row=5, column=1, padx=10, pady=3)
    button1.grid(row=4, column=1, padx=10, pady=3)
    button2.grid(row=3, column=1, padx=10, pady=3)
    label1.grid(row=1, column=2, padx=5, pady=1)
    label2.grid(row=2, column=2, padx=5, pady=1)
    root1.mainloop()
    mainwindows()


def search():
    global root
    root.quit()  # 结束主循环
    root.destroy()  # 销毁窗口
    root1 = tkinter.Tk()
    root1.title("c2")
    root1.geometry("720x720")
    root1["background"] = '#F8F8FF'

    # 填写函数区
    lb1 = tkinter.Label(root1, text='key1')
    lb1.place(relx=0.1, rely=0, relwidth=0.3, relheight=0.1)
    lb2 = tkinter.Label(root1, text='key2')
    lb2.place(relx=0.6, rely=0, relwidth=0.3, relheight=0.1)
    lb3 = tkinter.Label(root1, text='value')
    lb3.place(relx=0.1, rely=0.2, relwidth=0.3, relheight=0.1)
    lb4 = tkinter.Label(root1, text='k1_name')
    lb4.place(relx=0.6, rely=0.2, relwidth=0.3, relheight=0.1)
    inp1 = tkinter.Entry(root1)
    inp1.place(relx=0.1, rely=0.1, relwidth=0.3, relheight=0.1)
    inp2 = tkinter.Entry(root1)
    inp2.place(relx=0.6, rely=0.1, relwidth=0.3, relheight=0.1)
    inp3 = tkinter.Entry(root1)
    inp3.place(relx=0.1, rely=0.3, relwidth=0.3, relheight=0.1)
    inp4 = tkinter.Entry(root1)
    inp4.place(relx=0.6, rely=0.3, relwidth=0.3, relheight=0.1)
    btn1 = tkinter.Button(root1, text='打印结果', command=lambda: showe(inp1.get(), inp2.get(), inp3.get(), inp4.get()))
    btn1.place(relx=0.1, rely=0.45, relwidth=0.3, relheight=0.1)
    btn2 = tkinter.Button(root1, text='结果可视化', command=lambda: draw(inp1.get(), inp2.get(), inp3.get(), inp4.get()))
    btn2.place(relx=0.6, rely=0.45, relwidth=0.3, relheight=0.1)
    txt = tkinter.Text(root1)
    txt.place(rely=0.6, relheight=0.2, relwidth=1)

    def showe(k1, k2, v, n):
        if (v == 'Name'):
            txt.insert(tkinter.END, df.pivot_table(index=[k1, k2], values=v, aggfunc='count').loc[n, :].sort_values(v,
                                                                                                                    ascending=False).head())
        elif (v == 'Global_Sales'):
            txt.insert(tkinter.END, df.pivot_table(index=[k1, k2], values=v, aggfunc='sum').loc[n, :].sort_values(v,
                                                                                                                  ascending=False).head())
        elif (v == 'area'):
            txt.insert(tkinter.END, df.pivot_table(index=[k1],
                                                   values=['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales',
                                                           'Other_Sales'], aggfunc='sum').loc[n,
                                    :].sort_values().head())

    # inp1.delete(0,tkinter.END)

    def draw(k1, k2, v, n):
        root2 = tkinter.Tk()
        root2.title("c3")
        root2.geometry("980x720")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 6), sharex=True, sharey=True)
        if (v == 'Name'):
            df.pivot_table(index=[k1, k2], values=v, aggfunc='count').loc[n, :].sort_values(v,
                                                                                            ascending=False)[:20].plot.bar(
                ax=ax)
        elif (v == 'Global_Sales'):
            df.pivot_table(index=[k1, k2], values=v, aggfunc='sum').loc[n, :].sort_values(v, ascending=False)[
            :20].plot.bar(ax=ax)
        elif (v == 'area'):
            df.pivot_table(index=[k1], values=['Global_Sales', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
                           aggfunc='sum').loc[n, :].sort_values()[:20].plot.bar(ax=ax)
        canvas = FigureCanvasTkAgg(fig, master=root2)
        canvas.draw()  # 注意show方法已经过时了,这里改用draw
        canvas.get_tk_widget().pack(side=tkinter.TOP,  # 上对齐
                                    fill=tkinter.BOTH,  # 填充方式
                                    expand=tkinter.YES)  # 随窗口大小调整而调整

    def quit():
        root1.quit()  # 结束主循环
        root1.destroy()  # 销毁窗口
        #mainwindows()

    button0 = tkinter.Button(root1, text="退出", command=quit, width=15, height=3, bg='#F0FFFF', fg='#191970',
                             activeforeground='#F0F8FF', activebackground='#4682B4')
    button0.pack(side=tkinter.BOTTOM)
    root1.mainloop()
    mainwindows()


# 导入数据
plt.style.use('ggplot')  # 使用ggplot风格
na_values = ['N/A']  # 缺失值类型为N/A
df = pd.read_csv('vgsales.csv', na_values=na_values)  # 读入数据
df = df.dropna(how='any', axis=0)

# 创建主窗口
mainwindows()
