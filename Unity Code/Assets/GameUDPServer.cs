// -------------------------------------------------------------------------- //
// 					Bimeo Game UDP Server						 	  		  //
// -------------------------------------------------------------------------- //
//	File:			GameUDPServer.js							  		  	  //
//	Description:	Game UDP Server script for BiMeo Games		  	  		  //
// -------------------------------------------------------------------------- //
// 	Date created:	May 2013												  //	
//	Date modified:	July 2013												  //
// -------------------------------------------------------------------------- //
// 	Created by:		Matjaz Mihelj											  //	
//	Extended by:	Ales Hribar								 				  //
// -------------------------------------------------------------------------- //
//					All rights reserved Kinestica							  //
// -------------------------------------------------------------------------- //

using UnityEngine;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Text;
using System.IO;

public class GameUDPServer : MonoBehaviour
{

    // RECEIVE THREAD PARAMETERS
    const int PORT_RECEIVE = 25008;
    const int NUM_RECEIVE_FLOATS = 7;
    private Thread UDPThreadReceive;
    float[] dataReceive = new float[NUM_RECEIVE_FLOATS];
    bool isRunningReceive = true;

    // SEND THREAD PARAMETERS
    const int PORT_SEND = 25106;
    const int NUM_SEND_FLOATS = 7;
    private Thread UDPThreadSend;
    float[] dataSend = new float[NUM_SEND_FLOATS];
    bool isRunningSend = true;

    public bool showInput = true;
    //	private string IPSendAddress;	

    // DEBUG MODE PARAMETERS
    bool showDebug = false;
    private string udpStatus = "";

    //////////////////////////////////////////////////////
    // 	UPD SEND AND RECEIVE INITIALIZATION
    //////////////////////////////////////////////////////	
    void Start()
    {

        print("Start");
        try
        {
            //Starting the UDP Server thread.
            UDPThreadReceive = new Thread(new ThreadStart(StartReceive));
            UDPThreadSend = new Thread(new ThreadStart(StartSend));

            isRunningReceive = true;
            isRunningSend = true;

            UDPThreadReceive.IsBackground = true;
            UDPThreadSend.IsBackground = true;

            UDPThreadReceive.Start();
            UDPThreadSend.Start();

            udpStatus = "Started UDP Receiver and Send Threads!";
        }
        catch (Exception e)
        {
            udpStatus = "An UDP Exception has occurred: " + e.ToString();
        }
    }

    //////////////////////////////////////////////////////
    // 	STOP RECEIVING AND SENDING DATA WHEN APPLICATION QUIT
    //////////////////////////////////////////////////////	
    void OnApplicationQuit()
    {
        isRunningReceive = false;
        isRunningSend = false;
        udpStatus = "UDP Stopped.";
    }


    //////////////////////////////////////////////////////
    // 	RECEIVING DATA THROUGH UDP PORT
    //////////////////////////////////////////////////////		
    void StartReceive()
    {
        byte[] data = new byte[NUM_RECEIVE_FLOATS * 4];

        IPEndPoint ipep = new IPEndPoint(IPAddress.Any, PORT_RECEIVE);
        Socket newsock = new Socket(AddressFamily.InterNetwork,
                      SocketType.Dgram, ProtocolType.Udp);

        try
        {
            newsock.Bind(ipep);
            udpStatus = "Waiting for a client...";
            IPEndPoint sender = new IPEndPoint(IPAddress.Any, 0);
            EndPoint Remote = (EndPoint)(sender);

            try
            {
                newsock.ReceiveFrom(data, ref Remote);
                udpStatus = "Message received";

                while (isRunningReceive)
                {
                    data = new byte[1024];

                    try
                    {
                        newsock.ReceiveFrom(data, ref Remote);

                        // GET VALUES
                        for (int i = 0; i < NUM_RECEIVE_FLOATS; i++)
                        {
                            dataReceive[i] = BitConverter.ToSingle(data, sizeof(float) * i);
                            Console.WriteLine(dataReceive[i]);
                        }

                    }
                    catch (Exception e)
                    {
                        udpStatus = "An UDP Exception has occurred. Problem Receiving:  " + e.ToString();
                    }
                }

                newsock.Close();
            }
            catch (Exception e)
            {
                udpStatus = "An UDP Exception has occurred. Problem Receiving First:  " + e.ToString();
            }
        }
        catch (Exception e)
        {
            udpStatus = "An UDP Exception has occurred. Problem Connecting: " + e.ToString();
        }
    }

    //////////////////////////////////////////////////////
    // 	SENDING DATA THROUGH UDP PORT
    //////////////////////////////////////////////////////	

    void StartSend()
    {

        UdpClient sock = new UdpClient();
        IPEndPoint iep = new IPEndPoint(IPAddress.Parse("127.0.0.1"), PORT_SEND);

        byte[] dataB;
        dataB = new byte[NUM_SEND_FLOATS * sizeof(float)];

        while (isRunningSend)
        {
            Thread.Sleep(20);

            for (int i = 0; i < NUM_SEND_FLOATS; i++)
            {
                byte[] temp = BitConverter.GetBytes(dataSend[i]);

                for (int j = 0; j < sizeof(float); j++)
                {
                    dataB[i * sizeof(float) + j] = temp[j];
                }

            }

            try
            {
                sock.Send(dataB, dataB.Length, iep);

            }
            catch (Exception e)
            {
                udpStatus = "An UDP Exception has occurred. Problem Connecting: " + e.ToString();
            }
        }

        sock.Close();

    }

    //////////////////////////////////////////////////////
    // 	SET STATUS FOR RECEIVING AND SENDING DATA
    //////////////////////////////////////////////////////

    void SetIsRunningReceive(bool isRunReceive)
    {
        isRunningReceive = isRunReceive;
    }

    void SetIsRunningSend(bool isRunSend)
    {
        isRunningSend = isRunSend;
    }


    //////////////////////////////////////////////////////
    // 	CLOSE UDP CONNECTION
    //////////////////////////////////////////////////////	
    public void closeConnection()
    {
        isRunningReceive = false;
        isRunningSend = false;
    }


    //////////////////////////////////////////////////////
    // 	PARSE DATA FROM RECEIVE UDP PORT
    //////////////////////////////////////////////////////

    public float getDataX()
    {
        return dataReceive[0];
    }

    public float[] getQ_dot()
    {
        float[] q_dot = new float[7];
        for (int i = 0; i < 7; i++)
        {
            q_dot[i] = dataReceive[i];
        }
        return q_dot;
    }



    public void sendQ(float[] q)
    {
        for (int i = 0; i < 7; i++)
        {
            dataSend[i] = q[i];
        }
    }



    /*

	// SIMULINK COUNTER COUNTS WIHT 100 Hz FREQUENCY
	public float getMatlabCounter()
	{
		return dataReceive[0];
	}
	
	// HAND POSITION IN UNITY CS
	public Vector3 getHandPosition()
	{
		Vector3 tmpVec = new Vector3(dataReceive[1], dataReceive[2], dataReceive[3]);
		return tmpVec;
	}
	
	// HAND ORIENTATION IN UNITY CS
	public Quaternion getHandOrientation()
	{
		Quaternion tmpVec = new Quaternion(dataReceive[5], dataReceive[6], dataReceive[7], dataReceive[4]);
		return tmpVec;
	}	
	
	// GLOBAL FORCE IN UNITY CS
	public Vector3 getGlobalForce()
	{
		Vector3 tmpVec = new Vector3(dataReceive[8], dataReceive[9], dataReceive[10]);
		return tmpVec;
	}	
	
	// LOCAL FORCE IN HAND CS
	public Vector3 getLocalForce()
	{
		Vector3 tmpVec = new Vector3(dataReceive[11], dataReceive[12], dataReceive[13]);
		return tmpVec;
	}
	
	// SCALED GLOBAL FORCE IN UNITY CS
	public Vector3 getGlobalScaledForce()
	{
		Vector3 tmpVec = new Vector3(dataReceive[14], dataReceive[15], dataReceive[16]);
		return tmpVec;
	}	
	
	// SCALED LOCAL FORCE IN UNITY CS
	public Vector3 getLocalScaledForce()
	{
		Vector3 tmpVec = new Vector3(dataReceive[17], dataReceive[18], dataReceive[19]);
		return tmpVec;
	}	
	
	// ARM POWER GIVEN AS RGB COMPONENTS, RED WHEN PARETIC ARM SUPPORTED AND GREEN VICE VERSA
	public Vector3 getArmPowerRatio()
	{
		Vector3 tmpVec = new Vector3(dataReceive[20], dataReceive[21], dataReceive[22]);
		return tmpVec;
	}
	
	// SHOULDER JOINT ANGLES
	public Vector3 getShoulderAngles()
	{
		Vector3 tmpVec = new Vector3(dataReceive[23], dataReceive[24], dataReceive[25]);
		return tmpVec;
	}	
	
	// ELBOW JOINT ANGLE
	public float getElbowAngle()
	{
		return dataReceive[26];
	}	
	
	// WRIST JOINT ANGLES
	public Vector3 getWristAngles()
	{
		Vector3 tmpVec = new Vector3(dataReceive[27], dataReceive[28], dataReceive[29]);
		return tmpVec;
	}	
	
	// SHOULDER JOINT TORQUES
	public Vector3 getShoulderTorques()
	{
		Vector3 tmpVec = new Vector3(dataReceive[30], dataReceive[31], dataReceive[32]);
		return tmpVec;
	}	
	
	// ELBOW JOINT TORQUE
	public float getElbowTorque()
	{
		return dataReceive[33];
	}	
	
	// WRIST JOINT TORQUES
	public Vector3 getWristTorques()
	{
		Vector3 tmpVec = new Vector3(dataReceive[34], dataReceive[35], dataReceive[36]);
		return tmpVec;
	}		
	
	// DEVICE CONNECTED (RIGHT = 1 OR LEFT = 2)
	public int getDeviceConnected()
	{
		return (int) dataReceive[37];
		
	}	
	
	// BATTERY STATUS (BIMEO, FOREARM, UPPER ARM, TRUNK)
	public Vector4 getBatteryStatus()
	{
		Vector4 tmpVec = new Vector4(dataReceive[38], dataReceive[39], dataReceive[40], dataReceive[41]);
		return tmpVec;
	}	
	
	// DETECT SENSOR ERROR (FORCE OR IMU)
	public int getSensorError()
	{
		return (int) dataReceive[42];
		
	}		
//////////////////////////////////////////////////////
// 	COLLECT DATA FOR SEND UDP PORT
//////////////////////////////////////////////////////	
	
	// BIMEO INITIALIZATION
	public void startBimeoInitialization(float startInitialization)
	{
		dataSend[0] = startInitialization;
	}
	
	// SESSION TERMINATION
	public void terminateBimeoSession(float terminateSession)
	{
		dataSend[1] = terminateSession;
	}
	
	// PATIENT ANTHROPOMETRY
	public void setPatientAnthropometry(float patientHeight, float patientWeight)
	{
		dataSend[2] = patientHeight;
		dataSend[3] = patientWeight;
	}	
	
	// VARIOUS CONTROL SIGNALS
	public void setControlSignals(float controlSignal1, float controlSignal2, float controlSignal3, float controlSignal4, float controlSignal5)
	{
		dataSend[4] = controlSignal1;
		dataSend[5] = controlSignal2;
		dataSend[6] = controlSignal3;
		dataSend[7] = controlSignal4;
		dataSend[8] = controlSignal5;
	}		
	
	// SESSION FILE NAME TAG
	public void setNameTag(float rndNameTag)
	{
		dataSend[9] = rndNameTag;
	}
	
	// TRAINING MODE
	public void setTrainingMode(float manualMode, float planeMode, float jointOfInterest)
	{
		dataSend[10] = manualMode;
		dataSend[11] = planeMode;
		dataSend[12] = jointOfInterest;
	}
	
	// POSITION OFFSET
	public void setOffset(float centerPointPositionOffsetX, float centerPointPositionOffsetY, float centerPointPositionOffsetZ, float centerPointOrientationOffset)
	{
		dataSend[13] = centerPointPositionOffsetX;
		dataSend[14] = centerPointPositionOffsetY;
		dataSend[15] = centerPointPositionOffsetZ;
		dataSend[16] = centerPointOrientationOffset;
	}
	
	// MOTION SCALING
	public void setScaling(float scaleFactorX, float scaleFactorY)
	{
		dataSend[17] = scaleFactorX;
		dataSend[18] = scaleFactorY;
	}		
	
	// CURRENT SCENE (GAME)
	public void setScene(float currentSceneID)
	{
		dataSend[19] = currentSceneID;
	}
	
	// TARGET POSITION
	public void setTargetPosition(float targetPositionX,float targetPositionY,float targetPositionZ)
	{
		dataSend[20] = targetPositionX;
		dataSend[21] = targetPositionY;
		dataSend[22] = targetPositionZ;
	}
	
	// GAME STATE
	public void setGameState(float gameState)
	{
		dataSend[23] = gameState;
	}	
	
	
	// GAME SCORE
	public void setGameScore(float gameScore)
	{
		dataSend[24] = gameScore;
	}	
	
	// ADDITIONAL GAME VARIABLES
	public void setGameVariables(float gameVariable1, float gameVariable2, float gameVariable3, float gameVariable4, float gameVariable5)
	{
		dataSend[25] = gameVariable1;
		dataSend[26] = gameVariable2;
		dataSend[27] = gameVariable3;
		dataSend[28] = gameVariable4;
		dataSend[29] = gameVariable5;
	}	
	*/


    void OnGUI()
    {
        if (showDebug)
        {
            Rect rectObj = new Rect(20, Screen.height - 60, Screen.width - 40, 22);

            GUIStyle style = new GUIStyle();
            style.alignment = TextAnchor.UpperLeft;

            GUI.Box(rectObj, "UDP Status: " + udpStatus, style);
        }

        // TOGGLE DISPLAY
        if (Input.GetKey(KeyCode.F1))
        {
            showDebug = true;
        }

        if (Input.GetKey(KeyCode.C))
        {
            showDebug = false;
        }
    }
}