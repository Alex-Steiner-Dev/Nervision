using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using UnityEngine;
using UnityEngine.Networking;

public class ImageGeneration : MonoBehaviour
{
    [SerializeField] private string serverUrl;

    private void Start()
    {
        generateImage("test");
    }

    private void generateImage(string prompt)
    {
        using (var wb = new WebClient())
        {
            var response = wb.DownloadString(serverUrl);
            print(response);
        }
    }
}
