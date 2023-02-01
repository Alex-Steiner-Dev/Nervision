using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(ColorDetection))]
[RequireComponent(typeof(ComponentsDetection))]

public class Training : MonoBehaviour
{
    [SerializeField] private GameObject[] models;

    private ColorDetection colorDetection;
    private ComponentsDetection componentsDetection;

    private void Awake()
    {
        colorDetection = GetComponent<ColorDetection>();
        componentsDetection = GetComponent<ComponentsDetection>();
    }

    private void Start()
    {
        Train();
    }

    private void Train()
    {
        foreach(GameObject model in models)
        {
            Execute(model);
        }
    }

    private void Execute(GameObject model)
    {
        Color[] colors = colorDetection.DetectColor(model);

        foreach(Color color in colors)
        {
            Debug.Log(color);
        }
    }
}
