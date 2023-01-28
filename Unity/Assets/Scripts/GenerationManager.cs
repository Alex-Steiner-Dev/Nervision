using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GenerationManager : MonoBehaviour
{
    [HideInInspector] private Manager manager;

    private void Start()
    {
        manager = GameObject.FindGameObjectWithTag("GameController").GetComponent<Manager>();
    }

    public void Generate(string input)
    {
        manager.infoText.text = "Generating...";
    }
}
