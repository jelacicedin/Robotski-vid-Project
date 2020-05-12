using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(Camera))]
public class SnapshotCamera : MonoBehaviour
{
    Camera snapCam;

    int resWidth;
    int resHeight;

    int counter;

    void Start()
    {
        counter = 0;

        snapCam = new Camera();
        snapCam = Camera.allCameras[2];


         if (snapCam.targetTexture == null)
        {
            snapCam.targetTexture = new RenderTexture(resWidth, resHeight, 24);
        } else
        {
            resWidth = snapCam.targetTexture.width;
            resHeight = snapCam.targetTexture.height;
        }

    }


    void Update()
    {
        if (Input.GetKey(KeyCode.Space) && counter == 0)
        {
            counter = 10;

            snapCam.gameObject.SetActive(true);
            TakePic();
        }

        if (counter != 0)
        {
            counter--;
        }
        
    }

    void TakePic()
    {
        if (snapCam.gameObject.activeSelf)
        {
            Texture2D snapshot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
            snapCam.Render();

            RenderTexture.active = snapCam.targetTexture;

            snapshot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);

            byte[] bytes = snapshot.EncodeToPNG();
            string fileName = SnapshotName();
            System.IO.File.WriteAllBytes(fileName, bytes);

            Debug.Log("Snapshot from end effector camera taken");

            
        }
    }

    private string SnapshotName()
    {
        return string.Format("{0}/Snapshots/snap_{1}x{2}_{3}.png",
            Application.dataPath,
            resWidth,
            resHeight,
            DateTime.Now.ToString("yyyy-MM-dd_HH-mm-ss"));
    }
}
