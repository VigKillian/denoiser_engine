// test_couleur.cpp : Seuille une image en niveau de gris



#include <stdio.h>
#include <math.h>
#include "image_ppm.h"
#include <algorithm>
#include <iostream>
#include <random>

int main(int argc, char* argv[])
{
  char cNomImgLue[250], cNomImgEcrite[250];
  int nH, nW, nTaille;
  float pi = 3.14159f;
  float moyen = 0.f;float thita = 0.f;
  
  if (argc != 3) 
     {
       printf("Usage: ImageIn.pgm ImageOut.pgm\n"); 
       exit (1) ;
     }
   
   sscanf (argv[1],"%s",cNomImgLue) ;
   sscanf (argv[2],"%s",cNomImgEcrite);

   OCTET *ImgIn, *ImgOut;
   
   lire_nb_lignes_colonnes_image_pgm(cNomImgLue, &nH, &nW);
   nTaille = nH * nW;
  
   allocation_tableau(ImgIn, OCTET, nTaille);
   lire_image_pgm(cNomImgLue, ImgIn, nH * nW);
   allocation_tableau(ImgOut, OCTET, nTaille);
	
    double p = 0.05;                          // 5% des pixels bruités
    std::mt19937 rng(12345);
    std::uniform_real_distribution<double> u01(0.0, 1.0);

    for (int i = 0; i < nTaille; ++i) {
        double r = u01(rng);
        if (r < p) {
            ImgOut[i] = (r < p/2.0) ? 0 : 255;
        } else {
            ImgOut[i] = ImgIn[i];
        }
    }
   ecrire_image_pgm(cNomImgEcrite, ImgOut,  nH, nW);
   free(ImgIn); free(ImgOut);

   return 1;
}