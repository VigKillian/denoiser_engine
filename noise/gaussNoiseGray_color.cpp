// test_couleur.cpp : Seuille une image en niveau de gris

#include <stdio.h>
#include <math.h>
#include "image_ppm.h"
#include <algorithm>
#include <iostream>
#include <random>
/*
for f in input_images/*.pgm; do
  [ -e "$f" ] || continue 
  base="$(basename "${f%.*}")" 
  ./gaussNoiseGray "$f" "output_images/${base}_noise.pgm"
done
*/

int main(int argc, char* argv[])
{
  char cNomImgLue[250], cNomImgEcrite[250];
  int nH, nW, nTaille;
  float pi = 3.14159f;
  float moyenR = 0.f; float moyenG = 0.f; float moyenB = 0.f;
  float thitaR = 0.f; float thitaG = 0.f; float thitaB = 0.f;
  
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

	for(int i=0;i<nTaille3;i+=3){
		moyenR+= (float)ImgIn[i];
		moyenG+= (float)ImgIn[i+1];
		moyenB+= (float)ImgIn[i+2];
	}
	moyenR = moyenR / nTaille;
	moyenG = moyenG / nTaille;
	moyenB = moyenB / nTaille;
	// std::cout<<"moyen"<<moyen<<std::endl;

	for(int i=0;i<nTaille3;i+=3){
		thitaR += ((float)ImgIn[i]-moyenR)*((float)ImgIn[i]-moyenR)/nTaille;
		thitaG += ((float)ImgIn[i]-moyenG)*((float)ImgIn[i]-moyenG)/nTaille;
		thitaB += ((float)ImgIn[i]-moyenB)*((float)ImgIn[i]-moyenB)/nTaille;
	}
	thitaR = sqrt(thitaR);
	thitaG = sqrt(thitaG);
	thitaB = sqrt(thitaB);
	// std::cout<<"ecart type"<<thita<<std::endl;



	double sigma_bruit = 60.0;

	std::mt19937 rng(12345);
	std::normal_distribution<double> gauss(0.0, sigma_bruit);

	for (int i = 0; i < nTaille3; ++i) {
		double bruit = gauss(rng);
		int v = (int)lround((double)ImgIn[i] + bruit);
		v = std::clamp(v, 0, 255);
		ImgOut[i] = (OCTET)v;
	}

    ecrire_image_ppm(cNomImgEcrite, ImgOut,  nH, nW);
    free(ImgIn); free(ImgOut);

    return 1;
}