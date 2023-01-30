using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Net;
using System.Text;
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
        var wb = new WebClient();

        var data = new NameValueCollection();
        data["prompt"] = prompt;

        var response = wb.UploadValues(serverUrl, "POST", data);
        string responseInString = Encoding.UTF8.GetString(response);

        print(responseInString);
    }
}
