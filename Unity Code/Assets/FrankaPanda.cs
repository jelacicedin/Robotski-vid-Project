using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FrankaPanda : MonoBehaviour
{
    private GameObject J1;
    private GameObject J2;
    private GameObject J3;
    private GameObject J4;
    private GameObject J5;
    private GameObject J6;
    private GameObject J7;
    private GameObject UDP;
    private GameUDPServer UDPScript;
    public float q1_start = -60.0f, q2_start = -20.0f, q3_start = -50.0f, q4_start = -140.0f, q5_start = 20.0f, q6_start = 160.0f, q7_start = 0.0f;
    public float q_dotdot_multiplier = 1.0f;
    public float braking_multiplier = 1.0f;
    //float q_rotationSpeed = 40;
    float[] q;
    float[] q_dot;
    float[] q_dot_ref;
    float[] q_dotdot;
    float[] qMax;
    float[] qMin;
    float[] q_dotMax;
    public bool freespace = false;
    // Start is called before the first frame update
    void Start()
    {
        UDP = GameObject.Find("UDP");
        if (GameObject.Find("UDP"))
        {
            UDP = GameObject.Find("UDP");
            UDPScript = UDP.GetComponent<GameUDPServer>();
            print("UDP GameObject found");
        }
        J1 = GameObject.Find("Panda_J2");
        J2 = GameObject.Find("Panda_J3");
        J3 = GameObject.Find("Panda_J4");
        J4 = GameObject.Find("Panda_J5");
        J5 = GameObject.Find("Panda_J6");
        J6 = GameObject.Find("Panda_J7");
        J7 = GameObject.Find("Panda_TCP");
        q = new float[] { q1_start, q2_start, q3_start, q4_start, q5_start, q6_start, q7_start };
        q_dot = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        q_dot_ref = new float[] { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
        q_dotdot = new float[] { 859.4367f, 4297.1835f, 572.9578f, 7161.9724f, 859.4367f, 1145.9156f, 1445.9156f };
        qMax = new float[] { 166.003062f, 101.0010001f, 166.003062f, -3.99925f, 166.0031f, 215.0024f, 166.0031f };
        qMin = new float[] { -166.003062f, -101.0010001f, -166.003062f, -176.0012f, -166.0031f, -1.00268f, -166.0031f };
        q_dotMax = new float[] { 124.61832f, 124.61832f, 124.61832f, 124.61832f, 149.54198f, 149.54198f, 149.54198f };

        /* //initial values
        J1.transform.localEulerAngles = new Vector3(0, -90, -90);
        J2.transform.localEulerAngles = new Vector3(-180, 0, 0);
        J3.transform.localEulerAngles = new Vector3(90, 0, 0);
        J4.transform.localEulerAngles = new Vector3(45, 0, -180);
        J5.transform.localEulerAngles = new Vector3(-180, 0, 0);
        J6.transform.localEulerAngles = new Vector3(90, 90, 0);
        J7.transform.localEulerAngles = new Vector3(0, 0, 0);*/
    }

    // Update is called once per frame
    void Update()
    {
        //close the program on escape key
        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
        //check freespace
        if (Input.GetKeyDown(KeyCode.Return))
            freespace = !freespace;
        if (freespace)
        {
            //apply acceleration
            //joint 1
            if (Input.GetKey(KeyCode.Q)) q_dot_ref[0] = q_dotMax[0];
            else if (Input.GetKey(KeyCode.A)) q_dot_ref[0] -= q_dotMax[0];
            else q_dot_ref[0] = 0.0f;
            //joint 2
            if (Input.GetKey(KeyCode.W)) q_dot_ref[1] = q_dotMax[1];
            else if (Input.GetKey(KeyCode.S)) q_dot_ref[1] = -q_dotMax[1];
            else q_dot_ref[1] = 0.0f;
            //joint 3
            if (Input.GetKey(KeyCode.E)) q_dot_ref[2] = q_dotMax[2];
            else if (Input.GetKey(KeyCode.D)) q_dot_ref[2] = -q_dotMax[2];
            else q_dot_ref[2] = 0.0f;
            //joint 4
            if (Input.GetKey(KeyCode.R)) q_dot_ref[3] = q_dotMax[3];
            else if (Input.GetKey(KeyCode.F)) q_dot_ref[3] = -q_dotMax[3];
            else q_dot_ref[3] = 0.0f;
            //joint 5
            if (Input.GetKey(KeyCode.T)) q_dot_ref[4] = q_dotMax[4];
            else if (Input.GetKey(KeyCode.G)) q_dot_ref[4] = -q_dotMax[4];
            else q_dot_ref[4] = 0.0f;
            //joint 6
            if (Input.GetKey(KeyCode.Z)) q_dot_ref[5] = q_dotMax[5];
            else if (Input.GetKey(KeyCode.H)) q_dot_ref[5] = -q_dotMax[5];
            else q_dot_ref[5] = 0.0f;
            //joint 7
            if (Input.GetKey(KeyCode.U)) q_dot_ref[6] = q_dotMax[6];
            else if (Input.GetKey(KeyCode.J)) q_dot_ref[6] = -q_dotMax[6];
            else q_dot_ref[6] = 0.0f;


            //read inputs
            /*
            if (Input.GetKey(KeyCode.Q))q_dot[0] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.A)) q_dot[0] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[0] = 0.0f;
            if (Input.GetKey(KeyCode.W)) q_dot[1] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.S)) q_dot[1] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[1] = 0.0f;
            if (Input.GetKey(KeyCode.E)) q_dot[2] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.D)) q_dot[2] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[2] = 0.0f;
            if (Input.GetKey(KeyCode.R)) q_dot[3] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.F)) q_dot[3] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[3] = 0.0f;
            if (Input.GetKey(KeyCode.T)) q_dot[4] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.G)) q_dot[4] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[4] = 0.0f;
            if (Input.GetKey(KeyCode.Z)) q_dot[5] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.H)) q_dot[5] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[5] = 0.0f;
            if (Input.GetKey(KeyCode.U)) q_dot[6] = Time.deltaTime * q_rotationSpeed;
            else if (Input.GetKey(KeyCode.J)) q_dot[6] = -Time.deltaTime * q_rotationSpeed;
            else q_dot[6] = 0.0f;*/
        }
        //recieve q_dot_ref from simulink
        else
        {
            q_dot_ref = UDPScript.getQ_dot();
        }
        //limit q_dot_ref
        for (int i = 0; i < 7; i++)
        {

            if (q_dot_ref[i] > q_dotMax[i]) q_dot_ref[i] = q_dotMax[i];
            if (q_dot_ref[i] < -q_dotMax[i]) q_dot_ref[i] = -q_dotMax[i];
        }
        //q_dot regulator
        for (int i = 0; i < 7; i++)
        {
            if (q_dot[i] < q_dot_ref[i])
            {
                if (System.Math.Abs(q_dot_ref[i] - q_dot[i]) > (q_dotdot[i] * Time.deltaTime * q_dotdot_multiplier))
                    q_dot[i] += q_dotdot[i] * Time.deltaTime * q_dotdot_multiplier;
                else
                    q_dot[i] = q_dot_ref[i];
            }
            if (q_dot[i] > q_dot_ref[i])
            {
                if (System.Math.Abs(q_dot_ref[i] - q_dot[i]) > (q_dotdot[i] * Time.deltaTime * q_dotdot_multiplier))
                    q_dot[i] -= q_dotdot[i] * Time.deltaTime * q_dotdot_multiplier;
                else
                    q_dot[i] = q_dot_ref[i];
            }
        }
        //apply rotation
        for (int i = 0; i < 7; i++)
        {
            q[i] += q_dot[i] * Time.deltaTime;
            //limit qs
            if (q[i] > qMax[i])
            {
                q[i] = qMax[i];
                q_dot[i] = -q_dot[i] * 0.2f;
            }
            if (q[i] < qMin[i])
            {
                q[i] = qMin[i];
                q_dot[i] = -q_dot[i] * 0.2f;
            }
        }


        /*
        //apply rotation
            for (int i = 0; i < 7; i++)
            {
                if ((q_dot[i] > 0) & (q[i] + q_dot[i] > qMax[i]))
                {
                    q_dot[i] = qMax[i] - q[i];
                    q[i] = qMax[i];
                }
                else if ((q_dot[i] < 0) & (q[i] + q_dot[i] < qMin[i]))
                {
                    q_dot[i] = qMin[i] - q[i];
                    q[i] = qMin[i];
                }
                else
                    q[i] += q_dot[i];
            }*/
        J1.transform.localEulerAngles = new Vector3(-q[0], -90, -90);
        J2.transform.localEulerAngles = new Vector3(-180, 0, -q[1]);
        J3.transform.localEulerAngles = new Vector3(90, 0, 90 - q[2]);
        J4.transform.localEulerAngles = new Vector3(180 + q[3], 0, -180);
        J5.transform.localEulerAngles = new Vector3(-180, 0, 90 - q[4]);
        J6.transform.localEulerAngles = new Vector3(90, 90, 90 - q[5]);
        J7.transform.localEulerAngles = new Vector3(0, q[6], 0);
        UDPScript.sendQ(q);
    }

    void OnGUI()
    {
        GUIStyle style = new GUIStyle();
        style.fontSize = 24;
        for (int i = 0; i < 7; i++)
        {
            if ((System.Math.Round(q[i] - qMax[i]) == 0) | ((System.Math.Round(-q[i] + qMin[i]) == 0)))
                GUI.Label(new Rect(10, i * 30, 0, 0), "q" + (i + 1) + "= " + System.Math.Round(q[i], 2) + "°, Limit Reached", style);
            else
                GUI.Label(new Rect(10, i * 30, 0, 0), "q" + (i + 1) + "= " + System.Math.Round(q[i], 2) + "°", style);
        }
        if (freespace)
            GUI.Label(new Rect(10, 210, 0, 0), "Control: Keyboard", style);
        else
            GUI.Label(new Rect(10, 210, 0, 0), "Control: Simulink", style);
    }
}