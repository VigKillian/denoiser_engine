
#include <stdio.h>
#include "image_ppm.h"
#include <iostream>
#include <algorithm>
#include <random>

void rechercheMinMax(OCTET* ImgIn, int n, int& xminR, int& xminG, int& xminB, int& xmaxR, int& xmaxG, int& xmaxB){
    for(int i = 0; i<n; i++){
        if(xminR>ImgIn[3*i]) xminR = ImgIn[3*i];
        if(xminG>ImgIn[3*i+1]) xminG = ImgIn[3*i+1];
        if(xminB>ImgIn[3*i+2]) xminB = ImgIn[3*i+2];
        if(xmaxR<ImgIn[3*i]) xmaxR = ImgIn[3*i];
        if(xmaxG<ImgIn[3*i+1]) xmaxG = ImgIn[3*i+1];
        if(xmaxB<ImgIn[3*i+2]) xmaxB = ImgIn[3*i+2];
    }
}


int main(int argc, char* argv[])
{
  char cNomImgLue[250], cNomImgEcrite[250];
  int nH, nW, nTaille;
  
  if (argc != 3) 
     {
       printf("Usage: ImageIn.pgm ImageOut.pgm\n"); 
       exit (1) ;
     }
   
    sscanf (argv[1],"%s",cNomImgLue) ;
    sscanf (argv[2],"%s",cNomImgEcrite);

    OCTET *ImgIn, *ImgOut;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW;
    int nTaille3 = nTaille * 3;
    
    allocation_tableau(ImgIn, OCTET, nTaille3);
    lire_image_ppm(cNomImgLue, ImgIn, nH * nW);
    allocation_tableau(ImgOut, OCTET, nTaille3);

    for(int i = 0; i<nTaille3; i++){
        ImgOut[i] = (float)ImgIn[i] / 30.0;
    }

    int xminR = 255; int xminG = 255; int xminB = 255; int xmaxR = 0; int xmaxG = 0; int xmaxB = 0;

    rechercheMinMax(ImgOut, nTaille, xminR, xminG, xminB, xmaxR, xmaxG, xmaxB);

    // std::cout<<"Min R : "<< xminR << ", min G : " << xminG << ", min B : "<<xminB<<", max R : "<<xmaxR<<", max G : "<<xmaxG<<", max B : "<<xmaxB<<std::endl;

    double alphaR = -255*(double)xminR/(double)(xmaxR-xminR); double alphaG = -255*(double)xminG/(double)(xmaxG-xminG); double alphaB = -255*(double)xminB/(double)(xmaxB-xminB);
    double betaR = 255/(double)(xmaxR-xminR); double betaG = 255/(double)(xmaxG-xminG); double betaB = 255/(double)(xmaxB-xminB);

    // std::cout<<"Alpha R: "<<alphaR<<", alpha G : "<<alphaG<<", alpha B : "<<alphaB<<", beta R : "<<betaR<<", beta G : "<<betaG<<", beta B : "<<betaB<<std::endl;

    double amplitude = 1.0;                  // A : force du bruit (0..255)
	std::mt19937 rng(12345);                  // graine (fixe pour reproductible)
	std::uniform_real_distribution<double> uni(-amplitude, amplitude);

    for(int i = 0; i<nTaille3; i++){
        double bruit = uni(rng);              // U(-A, A)
		int v = (int)lround((double)ImgOut[i] + bruit);
		v = std::clamp(v, 0, 255);
		ImgOut[i] = (OCTET)v;
    }

    for(int i = 0; i<nTaille; i++){
        ImgOut[3*i] = (OCTET) alphaR + betaR*ImgOut[3*i];
        ImgOut[3*i+1] = (OCTET) alphaG + betaG*ImgOut[3*i+1];
        ImgOut[3*i+2] = (OCTET) alphaB + betaB*ImgOut[3*i+2];
    }
    

    
    ecrire_image_ppm(cNomImgEcrite, ImgOut,  nH, nW);
    free(ImgIn);
    free(ImgOut);
    return 1;
}
