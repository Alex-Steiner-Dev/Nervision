import numpy as np
from PIL import Image

def image_to_point_cloud(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)

    height, width = image_array.shape[:2]

    points = []

    for y in range(height):
        for x in range(width):
            pixel = image_array[y, x]
            points.append([x, y, pixel])

    with open("point_cloud.ply", "w") as file:

        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write("element vertex " + str(len(points)) + "\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")

        for point in points:
            file.write(str(point[0]) + " " + str(point[1]) + " " + str(point[2]) + "\n")

if __name__ == "__main__":
    image_to_point_cloud("test.jpeg")