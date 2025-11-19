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
	
     for(int i=0;i<nTaille;i++){
        moyen+= (float)ImgIn[i];
     }
     moyen = moyen / nTaille;
     std::cout<<"moyen"<<moyen<<std::endl;

     for(int i=0;i<nTaille;i++){
      thita += ((float)ImgIn[i]-moyen)*((float)ImgIn[i]-moyen)/nTaille;
     }
     thita = sqrt(thita);
      std::cout<<"ecart type"<<thita<<std::endl;



     double sigma_bruit = 10.0;

      std::mt19937 rng(12345);
      std::normal_distribution<double> gauss(0.0, sigma_bruit);

      for (int i = 0; i < nTaille; ++i) {
          double bruit = gauss(rng);
          int v = (int)lround((double)ImgIn[i] + bruit);
          v = std::clamp(v, 0, 255);
          ImgOut[i] = (OCTET)v;
      }

   ecrire_image_pgm(cNomImgEcrite, ImgOut,  nH, nW);
   free(ImgIn); free(ImgOut);

   return 1;
}