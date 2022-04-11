import matplotlib.pyplot as plt

x = [1,2,3,4,5,6]
y1 = [1,2,3,4,5,6]
y2 = [9,6,2,7,8,1]

plt.plot(x,y1, label='y1')
plt.plot(x,y2, label='y2')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()