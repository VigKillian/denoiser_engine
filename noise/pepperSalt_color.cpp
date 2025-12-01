// test_couleur.cpp : Seuille une image en niveau de gris



#include <stdio.h>
#include <math.h>
#include "image_ppm.h"
#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <cctype>

unsigned int extraireSeedDepuisNom(const std::string& nomFichier)
{
    std::string digits;
    for (char c : nomFichier)
    {
        if (std::isdigit(c))
            digits.push_back(c);
    }

    if (digits.empty())
        return 0; // Seed par défaut si aucun chiffre trouvé

    return std::stoul(digits);  // convertit la suite de chiffres en entier
}

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
	
	lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
	nTaille = nH * nW;
	int nTaille3 = nTaille * 3;
	
	allocation_tableau(ImgIn, OCTET, nTaille3);
	lire_image_ppm(cNomImgLue, ImgIn, nH * nW);
	allocation_tableau(ImgOut, OCTET, nTaille3);
		
	double p = 0.05;                          // 5% des pixels bruités
	std::string nom = cNomImgLue;
	unsigned int seed = extraireSeedDepuisNom(nom);
	std::mt19937 rng(seed);
	std::uniform_real_distribution<double> u01(0.0, 1.0);

	for (int i = 0; i < nTaille3; i+=3) {
		double r = u01(rng);
		if (r < p) {
			OCTET val = (r < p/2.0) ? 0 : 255;
			ImgOut[i] = val;
			ImgOut[i+1] = val;
			ImgOut[i+2] = val;
		} else {
			ImgOut[i] = ImgIn[i];
			ImgOut[i+1] = ImgIn[i+1];
			ImgOut[i+2] = ImgIn[i+2];
		}
	}
	ecrire_image_ppm(cNomImgEcrite, ImgOut,  nH, nW);
	free(ImgIn); free(ImgOut);

	return 1;
}