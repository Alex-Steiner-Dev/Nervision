using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace ImageToPointCloud
{
    class Program
    {
        static void Main(string[] args)
        {
            // Load the image
            Bitmap image = (Bitmap)Image.FromFile("depth.png");

            // Create an array to store the points
            Point3D[] points = new Point3D[image.Width * image.Height];

            // Loop through the image pixels
            int index = 0;
            for (int y = 0; y < image.Height; y++)
            {
                for (int x = 0; x < image.Width; x++)
                {
                    // Get the color of the current pixel
                    Color pixel = image.GetPixel(x, y);

                    // Add the pixel color to the point cloud as the Z value
                    points[index++] = new Point3D(x, y, pixel.R);
                }
            }

            // Save the point cloud to a file
            using (System.IO.StreamWriter file = new System.IO.StreamWriter("point_cloud.ply"))
            {
                // Write the PLY header
                file.WriteLine("ply");
                file.WriteLine("format ascii 1.0");
                file.WriteLine("element vertex " + points.Length);
                file.WriteLine("property float x");
                file.WriteLine("property float y");
                file.WriteLine("property float z");
                file.WriteLine("end_header");

                // Write the points
                foreach (Point3D point in points)
                {
                    file.WriteLine(point.X + " " + point.Y + " " + point.Z);
                }
            }
        }
    }

    struct Point3D
    {
        public float X;
        public float Y;
        public float Z;

        public Point3D(float x, float y, float z)
        {
            X = x;
            Y = y;
            Z = z;
        }
    }
}