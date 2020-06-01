using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraSwitch : MonoBehaviour
{
    private Camera[] cameras;
    private int frameUpdateCounter;

    private SnapshotCamera snapCam;

    // Start is called before the first frame update
    void Start()
    {
        cameras = Camera.allCameras;

        frameUpdateCounter = 0;

        cameras[1].enabled = false;
        cameras[0].enabled = true;
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.C) && frameUpdateCounter == 0)
        {
            cameras[0].enabled = !cameras[0].enabled;
            cameras[1].enabled = !cameras[1].enabled;

            frameUpdateCounter = 10;
        }

        if (frameUpdateCounter != 0)
        {
            frameUpdateCounter--;
        }
        
    }
}
