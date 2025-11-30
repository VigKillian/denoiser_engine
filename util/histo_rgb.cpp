#include <stdio.h>
#include "image_ppm.h"

int main(int argc, char* argv[])
{
    char cNomImgLue[250], cNomHisto[250];
    int nH, nW, nTaille, coligne, numColigne;
    //1 = ligne ; 2 = colonne
    
    if (argc != 2) 
        {
        printf("Usage: ImageIn.ppm\n"); 
        exit (1) ;
        }
    
    sscanf (argv[1],"%s",cNomImgLue);

    OCTET *ImgIn, *ImgOut;
    
    lire_nb_lignes_colonnes_image_ppm(cNomImgLue, &nH, &nW);
    nTaille = nH * nW * 3;
    
    allocation_tableau(ImgIn, OCTET, nTaille);
    lire_image_ppm(cNomImgLue, ImgIn, nH * nW);

    int histoR[256];
    int histoG[256];
    int histoB[256];
    for(int i=0; i<256; i++){
        histoR[i]=0;
        histoG[i]=0;
        histoB[i]=0;
    }

    for (int i=0; i < nH; i++)
        for (int j=0; j < nW; j++)
            {
                histoR[ImgIn[3*(i*nW+j)]] +=1;
                histoG[ImgIn[3*(i*nW+j)+1]] +=1;
                histoB[ImgIn[3*(i*nW+j)+2]] +=1;
            }
    
    for(int i=0; i<256; i++){
        printf("%d %d %d %d\n", i, histoR[i], histoG[i], histoB[i]);
    }
    
}