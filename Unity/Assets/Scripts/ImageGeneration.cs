using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Net;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class ImageGeneration : MonoBehaviour
{
    [SerializeField] private string serverUrl;
    [SerializeField] private Image img;

    [Obsolete]
    public void generateImage(string prompt)
    {
        var wb = new WebClient();

        var data = new NameValueCollection();
        data["prompt"] = prompt;

        var response = wb.UploadValues(serverUrl, "POST", data);
        string responseInString = Encoding.UTF8.GetString(response);

        StartCoroutine(setImage(responseInString));
    }

    [Obsolete]
    private IEnumerator setImage(string url)
    {
        url = url.Remove(0, 3);
        url = url.Remove(url.Length - 1, 1);
        print(url);
        WWW www = new WWW(url);
        yield return www;
        img.sprite = Sprite.Create(www.texture, new Rect(0, 0, www.texture.width, www.texture.height), new Vector2(0, 0));
    }
}
