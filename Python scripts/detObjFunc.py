#from RVdolinar module from labs
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from scipy.interpolate import interpn
import math
def showImage(iImage, iTitle):
    plt.figure(iTitle)
    plt.imshow(iImage,cmap='gray',clim=[0,255])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
def loadImage(iPath):
    tempImage=Image.open(iPath)
    oImage=np.array(tempImage)
    return oImage
def saveImage(iPath, iImage, iFormat):
    tempI=Image.fromarray(iImage)
    tempI.save(iPath,format=iFormat)
def cropImage(iImage,x1,y1,x2,y2):
    # (x1,y1) zgornjo levo oglišče, (x2,y2) spodnjo desno oglišče
    oImage=iImage[y1:y2,x1:x2]
    return oImage
def mirrorVertical(iImage):
    yDim=np.shape(slika)[0]
    xDim=np.shape(slika)[1]
    oImage=np.zeros([yDim,xDim,3],dtype='uint8')
    x=0
    while x<xDim:
        oImage[:,xDim-x-1]=iImage[:,x]
        x+=1
    return oImage
def RGBtoGrayscale(iImage,iScale=(0.333,0.333,0.333)):
    y=0
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    tImage=np.zeros([yDim,xDim],dtype='float32')
    while y<yDim:
        x=0
        while x<xDim:
            tImage[y,x]=round(iImage[y,x,0]*iScale[0]+iImage[y,x,1]*iScale[1]+iImage[y,x,2]*iScale[2])
            if 255<round(iImage[y,x,0]*iScale[0]+iImage[y,x,1]*iScale[1]+iImage[y,x,2]*iScale[2]):
                tImage[y,x]=255
            if 0>round(iImage[y,x,0]*iScale[0]+iImage[y,x,1]*iScale[1]+iImage[y,x,2]*iScale[2]):
                tImage[y,x]=0
            x+=1
        y+=1
    oImage=tImage.astype('uint8')
    return oImage
def scaleImage(iImage,iSlopeA,iIntersectionB):
    y=0
    if (iImage.dtype=='uint8'):
        maxVal=255.0
    elif ((iImage.dtype=='float32')|(iImage.dtype=='float64')):
        maxVal=1.0
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    oImage=np.zeros_like(iImage)
    oImage.astype(iImage.dtype)
    errorcount=0
    #funkcija ni prilagojena za ne uint8 datatype!
    while y<yDim:
        x=0
        while x<xDim:
            oImage[y,x]=(iImage[y,x])*iSlopeA+iIntersectionB
            if (iImage[y,x])*iSlopeA+iIntersectionB>maxVal:
                oImage[y,x]=maxVal
                if errorcount==0:
                    print("scaleImage value over max")
                    errorcount=1
            if (iImage[y,x])*iSlopeA+iIntersectionB<0:
                oImage[y,x]=0
                if errorcount==0:
                    print("scaleImage value under 0")
                    errorcount=1
            x+=1
        y+=1
    return oImage
def windowImage(iImage,iCenter,iWidth):
    y=0
    if (iImage.dtype=='uint8'):
        maxVal=255.0
    elif ((iImage.dtype=='float32')|(iImage.dtype=='float64')):
        maxVal=1.0
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    oImage=np.zeros_like(iImage)
    oImage.astype(iImage.dtype)
    #funkcija ni prilagojena za ne uint8 datatype!
    errorcount=0
    while y<yDim:
        x=0
        while x<xDim:
            oImage[y,x]=(iImage[y,x]-iCenter+iWidth/2)*maxVal/iWidth
            #print(oImage[y,x])
            if (iImage[y,x]-iCenter+iWidth/2)*maxVal/iWidth>maxVal:
                oImage[y,x]=maxVal
                if errorcount==0:
                    print("windowImage value over max")
                    errorcount=1
            if (iImage[y,x]-iCenter+iWidth/2)*maxVal/iWidth<0:
                oImage[y,x]=0
                if errorcount==0:
                    print("windowImage value under 0")
                    errorcount=1
            x+=1
        y+=1
    return oImage
def thresholdImage(iImage,iThreshold):
    if (iImage.dtype=='uint8'):
        maxVal=255.0
    elif ((iImage.dtype=='float32')|(iImage.dtype=='float64')):
        maxVal=1.0
    y=0
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    oImage=np.zeros_like(iImage)
    oImage.astype(iImage.dtype)
    #funkcija ni prilagojena za ne uint8 datatype!
    while y<yDim:
        x=0
        while x<xDim:
            if iImage[y,x]>=iThreshold:
                oImage[y,x]=maxVal
            else:
                oImage[y,x]=0

            x+=1
        y+=1
    return oImage
def gammaImage(iImage,iGamma):
    y=0
    if (iImage.dtype=='uint8'):
        maxVal=255.0
    elif ((iImage.dtype=='float32')|(iImage.dtype=='float64')):
        maxVal=1.0
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    oImage=np.zeros_like(iImage)
    oImage.astype(iImage.dtype)
    #funkcija ni prilagojena za ne uint8 datatype!
    while y<yDim:
        x=0
        while x<xDim:
            oImage[y,x]=maxVal*(iImage[y,x]/maxVal)**iGamma
            x+=1
        y+=1
    return oImage
def convertImageColorSpace(iImage, iConversionType):
    iImage = np.array(iImage, dtype = 'float')
    oImage = np.zeros_like(iImage)
    if iConversionType == 'RGBtoHSV':
        r,g,b = iImage[:,:,0],iImage[:,:,1],iImage[:,:,2]
        r,g,b = r/255.0, g/255.0, b/255.0
        h,s,v = np.zeros_like(r),np.zeros_like(r),np.zeros_like(r)
        Cmax = np.maximum(r,np.maximum(g,b))
        Cmin = np.minimum(r,np.minimum(g,b))
        delta = Cmax - Cmin + 1e-7
        #1e-7 za float error
        h[Cmax == r] = 60.00 * (((g[Cmax == r]-b[Cmax==r])/delta[Cmax == r])%6)
        h[Cmax == g] = 60.00 * (((b[Cmax == g]-r[Cmax==g])/delta[Cmax == g])+2)
        h[Cmax == b] = 60.00 * (((r[Cmax == b]-g[Cmax==b])/delta[Cmax == b])+4)
        s = delta / Cmax
        v = Cmax
        oImage[:,:,0]=h
        oImage[:,:,1]=s
        oImage[:,:,2]=v
    if iConversionType == 'HSVtoRGB':
        h,s,v = iImage[:,:,0],iImage[:,:,1],iImage[:,:,2]
        C,X,m=np.zeros_like(h),np.zeros_like(h),np.zeros_like(h)
        r,g,b=np.zeros_like(h),np.zeros_like(h),np.zeros_like(h)
        C=v*s
        X=C*(1-abs(((h/60)%2)-1))
        m=v-C
        r[(0<=h) & (h<60)]=C[(0<=h) & (h<60)]
        g[(0<=h) & (h<60)]=X[(0<=h) & (h<60)]
        b[(0<=h) & (h<60)]=0
        r[(60<=h) & (h<120)]=X[(60<=h) & (h<120)]
        g[(60<=h) & (h<120)]=C[(60<=h) & (h<120)]
        b[(60<=h) & (h<120)]=0
        r[(120<=h) & (h<180)]=0
        g[(120<=h) & (h<180)]=C[(120<=h) & (h<180)]
        b[(120<=h) & (h<180)]=X[(120<=h) & (h<180)]
        r[(180<=h) & (h<240)]=0
        g[(180<=h) & (h<240)]=X[(180<=h) & (h<240)]
        b[(180<=h) & (h<240)]=C[(180<=h) & (h<240)]
        r[(240<=h) & (h<300)]=X[(240<=h) & (h<300)]
        g[(240<=h) & (h<300)]=0
        b[(240<=h) & (h<300)]=C[(240<=h) & (h<300)]
        r[(300<=h) & (h<360)]=C[(300<=h) & (h<360)]
        g[(300<=h) & (h<360)]=0
        b[(300<=h) & (h<360)]=X[(300<=h) & (h<360)]
        oImage[:,:,0]=r+m
        oImage[:,:,1]=g+m
        oImage[:,:,2]=b+m
    return oImage
def discreteConvolution2D(iImage, iKernel):
    yDim=np.shape(iImage)[0]
    xDim=np.shape(iImage)[1]
    extend=iKernel.shape[0]//2
    tmpImage=np.zeros((yDim+2*extend,xDim+2*extend),dtype='float')
    tmpImage[extend:-extend,extend:-extend]=iImage
    oImage=tmpImage[extend:-extend,extend:-extend]
    y=0
    while y<yDim:
        x=0
        while x<xDim:
            oImage[y,x]=round(np.multiply(tmpImage[y:y+1+2*extend,x:x+1+2*extend],iKernel).sum())
            x+=1
        y+=1
        oImage=oImage.astype(iImage.dtype)
    return oImage
def interpolate0Image2D(iImage,iCoorX,iCoorY):
    tmp=iImage[-1,:]
    tmp = np.reshape(tmp, (1, -1))
    iImage=np.concatenate((iImage,tmp),0)
    tmp=iImage[:,-1]
    tmp = np.reshape(tmp, (-1,1))
    iImage=np.concatenate((iImage,tmp),1)
    xDim=iCoorX.shape[1]
    yDim=iCoorX.shape[0]
    oImage=np.zeros((yDim,xDim),dtype=iImage.dtype)
    iCoorX=np.floor(iCoorX).astype(int)
    iCoorY=np.floor(iCoorY).astype(int)
    for y in range(yDim-1):
        for x in range(xDim-1):
            cy=iCoorY[y,x]
            cx=iCoorX[y,x]
            oImage[y,x]=iImage[cy,cx]
    return oImage
def interpolate1Image2D(iImage,iCoorX,iCoorY):
    xDim=iCoorX.shape[1]
    yDim=iCoorX.shape[0]
    tmp=iImage[-2::,:]
    tmp = np.reshape(tmp, (2, -1))
    iImage=np.concatenate((iImage,tmp),0)
    tmp=iImage[:,-2::]
    tmp = np.reshape(tmp, (-1,2))
    iImage=np.concatenate((iImage,tmp),1)
    oImage=np.zeros((yDim,xDim),dtype=iImage.dtype)
    for y in range(yDim):
        for x in range(xDim):
            cy=iCoorY[y,x]
            cx=iCoorX[y,x]
            fy=math.floor(cy)
            fx=math.floor(cx)
            a=(fx+1-cx)*(fy+1-cy)
            b=(cx-fx)*(fy+1-cy)
            c=(fx+1-cx)*(cy-fy)
            d=(cx-fx)*(cy-fy)
            oImage[y,x]=round(a*iImage[fy,fx]+c*iImage[fy+1,fx]+b*iImage[fy,fx+1]+d*iImage[fy+1,fx+1])
    return oImage
def interpolateColorImage( iImage, iCoorX, iCoorY, method):
    r = iImage[:,:,0]
    g = iImage[:,:,1]
    b = iImage[:,:,2]
    koord = iCoorX, iCoorY
    koblika = iCoorX.shape + (3,)
    oImage = np.zeros(koblika, dtype = 'uint8')
    y_s=iImage.shape[1]
    x_s=iImage.shape[0]
    x = np.arange(x_s)
    y = np.arange(y_s)
    if method == 'nearest':
        red = interpn((x, y), r, koord, 'nearest')
        green = interpn((x, y), g, koord, 'nearest')
        blue = interpn((x, y),b, koord,'nearest')
        oImage[:,:,0]=red
        oImage[:,:,1]=green
        oImage[:,:,2]=blue
        oImage = np.transpose(oImage,(1, 0, 2))
        return oImage
    if method == 'linear':
        red = interpn((x, y), r, koord, 'linear')
        green = interpn((x, y), g, koord,'linear')
        blue = interpn((x, y),b, koord, 'linear')
        oImage[:,:,0]=red
        oImage[:,:,1]=green
        oImage[:,:,2]=blue
        oImage = np.transpose(oImage,(1, 0, 2))
    return oImage
def decimateImage2D(iImage, iLevel):
    kernel=np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
    for i in range(iLevel):
        iImage=discreteConvolution2D(iImage,kernel)
        xDim=iImage.shape[1]
        yDim=iImage.shape[0]
        oImage=np.zeros((math.ceil(yDim/2),math.ceil(xDim/2)))
        for y in range(oImage.shape[0]):
            for x in range(oImage.shape[1]):
                oImage[y,x]=iImage[2*y,2*x]
        iImage=oImage
    return oImage
def plotDots(iPoints1,iPoints2,iPoints3=np.zeros(1),iLabel1="val 1",iLabel2="val 2", iLabel3="val 3"):
    plt.plot(iPoints1[:,0],iPoints1[:,1],'ob',markersize=5.0,label=iLabel1)
    plt.plot(iPoints2[:,0],iPoints2[:,1],'og',markersize=5.0,label=iLabel2)
    if not (iPoints3.size==1):
        plt.plot(iPoints3[:,0],iPoints3[:,1],'or',markersize=5.0,label=iLabel3)
def transAffine2D(iScale=(1, 1), iTrans=(0, 0), iRot=0, iShear=(0, 0)):
    iRot=iRot*np.pi/180
    Mscale=np.array(((iScale[0],0,0),(0,iScale[1],0),(0,0,1)))
    Mtrans=np.array(((1,0,iTrans[0]),(0,1,iTrans[1]),(0,0,1)))
    Mrot=np.array(((np.cos(iRot),-np.sin(iRot),0),(np.sin(iRot),np.cos(iRot),0),(0,0,1)))
    Mshear=np.array(((1,iShear[0],0),(iShear[1],1,0),(0,0,1)))
    oMat2D=np.dot(Mtrans, np.dot(Mshear, np.dot(Mrot,Mscale)))
    return oMat2D
def addHomCoord2D(iPts):
    if iPts.shape[-1] == 3:
        return iPts
    oPts = np.hstack((iPts, np.ones((iPts.shape[0], 1))))
    return oPts
def mapAffineApprox2D(iPtsRef, iPtsMov):
    """Afina aproksimacijska poravnava"""
    iPtsRef = np.matrix(iPtsRef)
    iPtsMov = np.matrix(iPtsMov)
    # po potrebi dodaj homogeno koordinato
    iPtsRef = addHomCoord2D(iPtsRef)
    iPtsMov = addHomCoord2D(iPtsMov)
    # afina aproksimacija (s psevdoinverzom)
    iPtsRef = iPtsRef.transpose()
    iPtsMov = iPtsMov.transpose()
    #print(iPtsRef.shape)
    #print(iPtsMov.shape)
    # psevdoinverz
    oMat2D = np.dot(iPtsRef, np.linalg.pinv(iPtsMov))
    # psevdoinverz na dolgo in siroko:
    #oMat2D = iPtsRef * iPtsMov.transpose() * \
    #np.linalg.inv( iPtsMov * iPtsMov.transpose() )
    return oMat2D
def findCorrespondingPoints(iPtsRef, iPtsMov):
    """Poisci korespondence kot najblizje tocke"""
    # inicializiraj polje indeksov
    iPtsMov = np.array(iPtsMov)
    iPtsRef = np.array(iPtsRef)

    idxPair = -np.ones((iPtsRef.shape[0], 1), dtype='int32')
    idxDist = np.ones((iPtsRef.shape[0], iPtsMov.shape[0]))
    for i in range(iPtsRef.shape[0]):
        for j in range(iPtsMov.shape[0]):
            idxDist[i,j] = np.sum((iPtsRef[i,:2] - iPtsMov[j,:2])**2)
    # doloci bijektivno preslikavo
    #print('idxDist dim: '+str(idxDist.shape))
    while not np.all(idxDist==np.inf):
        i, j = np.where(idxDist == np.min(idxDist))

        idxPair[i[0]] = j[0]
        idxDist[i[0],:] = np.inf
        idxDist[:,j[0]] = np.inf
        #print('(i,j): ('+str(i)+', '+str(j)+')\nidxPairt: '+str(idxPair.transpose())+'\nidxDist: '+str(idxDist))
    # doloci pare tock
    idxValid, idxNotValid = np.where(idxPair>=0)
    idxValid = np.array( idxValid )
    iPtsRef_t = iPtsRef[idxValid,:]
    iPtsMov_t = iPtsMov[idxPair[idxValid].flatten(),:]
    return iPtsRef_t, iPtsMov_t
def alignICP(iPtsRef, iPtsMov, iEps=1e-6, iMaxIter=50, plotProgress=False):
    """Postopek iterativno najblizje tocke"""
    # inicializiraj izhodne parametre
    curMat = []; oErr = []; iCurIter = 0
    if plotProgress:
        iPtsMov0 = np.matrix(iPtsMov)
        fig = plt.figure()
        ax = fig.add_subplot(111)
    # zacni iterativni postopek
    while True:
        # poisci korespondencne pare tock
        iPtsRef_t, iPtsMov_t = findCorrespondingPoints(iPtsRef, iPtsMov)
        # doloci afino aproksimacijsko preslikavo
        oMat2D = mapAffineApprox2D(iPtsRef_t, iPtsMov_t)
        # posodobi premicne tocke
        iPtsMov = np.dot(addHomCoord2D(iPtsMov), oMat2D.transpose())
        # izracunaj napako
        curMat.append(oMat2D)
        oErr.append(np.sqrt(np.sum((iPtsRef_t[:,:2]- iPtsMov_t[:,:2])**2)))
        iCurIter = iCurIter + 1
        # preveri kontrolne parametre
        dMat = np.abs(oMat2D - transAffine2D())
        if iCurIter>iMaxIter or np.all(dMat<iEps):
            break
    # doloci kompozitum preslikav
    oMat2D = transAffine2D()
    for i in range(len(curMat)):

        if plotProgress:
            iPtsMov_t = np.dot(addHomCoord2D(iPtsMov0), oMat2D.transpose())
            ax.clear()
            ax.plot(iPtsRef[:,0], iPtsRef[:,1], 'ob')
            ax.plot(iPtsMov_t[:,0], iPtsMov_t[:,1], 'om')
            fig.canvas.draw()
            plt.pause(0.1)

        oMat2D = np.dot(curMat[i], oMat2D)
    return oMat2D, oErr

    #end of original RVdolinar module contents
