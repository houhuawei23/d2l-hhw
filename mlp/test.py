import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def generate_test_photo(width, height, background_color=(255, 255, 255)):
    # 创建一个新的图像对象
    image = Image.new('RGB', (width, height), color=background_color)
    
    # 在图像上绘制一些简单的形状和文本（这里只是一个示例，你可以根据需要进行更复杂的绘制）
    draw = ImageDraw.Draw(image)
    
    # 绘制一个矩形
    rectangle_color = (0, 0, 255)
    rectangle_coords = [(50, 50), (width - 50, height - 50)]
    draw.rectangle(rectangle_coords, fill=rectangle_color, outline=None)
    
    # 绘制一个圆
    circle_color = (255, 0, 0)
    circle_center = (width // 2, height // 2)
    circle_radius = 100
    draw.ellipse((circle_center[0] - circle_radius, circle_center[1] - circle_radius,
                  circle_center[0] + circle_radius, circle_center[1] + circle_radius),
                 fill=circle_color, outline=None)
    
    # 添加一些文本
    text = "Test Photo"
    text_color = (0, 0, 0)
    text_position = (width // 4, height - 100)
    draw.text(text_position, text, fill=text_color)
    
    # 保存图像
    image.save("test_photo.jpg")

if __name__ == "__main__":
    width, height = 800, 600
    generate_test_photo(width, height)

img = Image.open('test_photo.jpg')
plt.imshow(img)
plt.show()
