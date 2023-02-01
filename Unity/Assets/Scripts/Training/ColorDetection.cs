using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ColorDetection : MonoBehaviour
{
    public Color[] DetectColor(GameObject mesh)
    {
        Color[] colors = new Color[0];
        Mesh _mesh = mesh.GetComponent<MeshFilter>().mesh;
        Vector3[] vertices = _mesh.vertices;

        colors = new Color[vertices.Length];

        for (int i = 0; i < vertices.Length; i++)
            colors[i] = Color.Lerp(Color.red, Color.green, vertices[i].y);

        _mesh.colors = colors;

        return colors;
    }
}
