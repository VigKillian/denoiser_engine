// test_couleur.cpp : Seuille une image en niveau de gris

#include "image_ppm.h"
#include <stdio.h>
#include <iostream>

using namespace std;

int main(int argc, char *argv[]) {
	char cNomImgLue[250], cNomFichierSortie[250],cNomImgEcrite[250];
	int nH, nW, nTaille;
    int nR,nG,nB;

	if (argc != 3) {
		printf("Usage: ImageIn.pgm ImageOut.pgm \n");
		exit(1);
	}

	sscanf(argv[1], "%s", cNomImgLue);
	sscanf(argv[2], "%s", cNomImgEcrite);


	OCTET *ImgIn, *ImgOut;

   lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
   nTaille = nH * nW;
  
   int nTaille3 = nTaille * 3;
   allocation_tableau(ImgIn, OCTET, nTaille3);
   lire_image_ppm(cNomImgLue, ImgIn, nH * nW);

   allocation_tableau(ImgOut, OCTET, nTaille3);

	ecrire_image_ppm(cNomImgEcrite, ImgOut,  nH, nW);
    
	return 1;
}
/*
 plot "armas_seu_extr_exp.dat" using 1:2 with lines title 'occurences de rouge ' lt rgb 'red' lw 2,\
''using 1:3 with lines title 'occurences de vert' lt rgb 'green' lw 2,\
''using 1:4 with lines title 'occurences de bleu' lt rgb 'blue'lw 2 三条一起出现

*/