"""
matplotlib:
일반적으로 많이 쓰이는 가상화 모듈
- pyplot 객체를 사용하여 데이터를 표시
- pyplot 객체에 그래프들을 쌓은 다음 show로 flush
- 여러개의 그림판을 만들 수 있음
- linestyle: 라인 스타일
- color: 색상
- 수식 사용 가능
- loc: location
- Scatter? : 산포드를 그릴 수 있음
- marker: 원하는 마크 생성
- s: 데이터의 크기를 지정, 데이터의 크기 비교 가능
- bar: 바 차트 사용
- bins: 나누는 개수
-
"""

import matplotlib.pyplot as plt
X = range(100)
Y = [value**2 for value in X]
plt.plot(X,Y)
plt.show()
import numpy as np

X_1 = range(100)
Y_1 = [np.cos(value) for value in X]

X_2 = range(100)
Y_2 = [np.sin(value) for value in X]

plt.plot(X_1, Y_1)
plt.plot(X_2, Y_2)
plt.plot(range(100), range(100))
plt.show()

fig = plt.figure() # figure 반환
fig.set_size_inches(10,10) # 크기지정
ax_1 = fig.add_subplot(1,2,1) # 두개의 plot 생성
ax_2 = fig.add_subplot(1,2,2)  # 두개의 plot 생성

ax_1.plot(X_1, Y_1, c="b")  # 첫번째 plot
ax_2.plot(X_2, Y_2, c="g")  # 두번째 plot
plt.show() # show & flush2)
plt.plot(range(100), range(100))
plt.show()
#Set color
X_1 = range(100)
Y_1 = [value for value in X]

X_2 = range(100)
Y_2 = [value + 100 for value in X]

plt.plot(X_1, Y_1, color="#000000")
plt.plot(X_2, Y_2, c="c")

plt.show()
#Set linestyle
plt.plot(X_1, Y_1, c="b", linestyle="dashed")
plt.plot(X_2, Y_2, c="r", ls="dotted")

plt.title("Two lines")
plt.show()
plt.plot(X_1, Y_1, color="b", linestyle="dashed")
plt.plot(X_2, Y_2, color="r", linestyle="dotted")

plt.title('$y = \\frac{ax + b}{test}$')
plt.show()

plt.plot(X_1, Y_1, color="b", linestyle="dashed")
plt.plot(X_2, Y_2, color="r", linestyle="dotted")

plt.text(50, 70, "Line_1")
plt.annotate(
    'line_2', xy=(50, 150), xytext=(20, 175),
    arrowprops=dict(facecolor='black', shrink=0.05))

plt.title('$y = ax+b$')
plt.xlabel('$x_line$')
plt.ylabel('y_line')

plt.show()
plt.plot(X_1, Y_1, color="b", linestyle="dashed", label='line_1')
plt.plot(X_2, Y_2, color="r", linestyle="dotted", label='line_2')
plt.legend(shadow=True, fancybox=False, loc="upper right")

plt.title('$y = ax+b$')
plt.xlabel('$x_line$')
plt.ylabel('y_line')


plt.show()
plt.plot(X_1, Y_1, color="b", linestyle="dashed", label='line_1')
plt.plot(X_2, Y_2, color="r", linestyle="dotted", label='line_2')
plt.legend(shadow=True, fancybox=True, loc="lower right")


plt.grid(True)
plt.xlim(-1000, 2000)
plt.ylim(-1000, 2000)

plt.show()
plt.plot(X_1, Y_1, color="b", linestyle="dashed", label='line_1')
plt.plot(X_2, Y_2, color="r", linestyle="dotted", label='line_2')

plt.grid(True, lw=0.4, ls="--", c=".90")
plt.legend(shadow=True, fancybox=True, loc="lower right")
plt.xlim(-100, 200)
plt.ylim(-200, 200)
plt.savefig("test.png", c="a") #파일 저장
plt.show()

#Scatter
data_1 = np.random.rand(512, 2)
data_2 = np.random.rand(512, 2)
plt.scatter(data_1[:,0], data_1[:,1], c="b", marker="x")
plt.scatter(data_2[:,0], data_2[:,1], c="r", marker="o")

plt.show()
N = 50
x = np.random.rand(N)
y = np.random.rand(N)
colors = np.random.rand(N)
area = np.pi * (15 * np.random.rand(N))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.show()
# Bar chart
data = [[5., 25., 50., 20.],
        [4., 23., 51., 17],
        [6., 22., 52., 19]]

X = np.arange(0,8,2)
plt.bar(X + 0.00, data[0], color = 'b', width = 0.50)
plt.bar(X + 0.50, data[1], color = 'g', width = 0.50)
plt.bar(X + 1.0, data[2], color = 'r', width = 0.50)
plt.xticks(X+0.50, ("A","B","C", "D"))
plt.show()
