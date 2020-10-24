#include <iostream>
#include<math.h>
using namespace std;

class Matrix
{
	public:
		//Data Members
		int r; //number of rows
		int c; //number of columns
		float m[100][100]; //matrix

		//Constructor
		Matrix(int row,int column);

		//Utility functions
		void printMatrix();

		//Member functions
		Matrix addition(); //Matrix addition
		Matrix subtraction(); //Matrix subtraction
		Matrix multiplication(); //Matrix multiplication
		int isIdentity(); //Returns 1 if it is an identity matrix else returns 0
		int isSquare(); //Checks if matrix is a square matrix
		int trace(); //Find the trace of a given matrix
		int *dimensions(); //Find the dimensions of a matrix
		int *gaussElimination();
		double determinant(int , double [10][10]);
		Matrix columnSpace();
		Matrix transpose();
		Matrix nullSpace();
		Matrix inverse();
		int *eigenValues();
		Matrix eigenVectors();
		int *graph(); //If the matrix represents a graph return number of edges and vertices else return NULL
		Matrix *LUdecomposition();
		Matrix *LDUdecomposition();
		Matrix *SVD(); //Single-value decomposition
		int determinant(); //Finds the determinant of the given matrix
		Matrix adjoint(); //Finds the adjoint of the given matrix
		int isInvertible(); //Returns 1 if the matrix is invertible else returns 0
		Matrix scalarMul(); //Multiplies the matrix by a scalar value
		int isIdempotent(); //Returns 1 if the matrix is idempotent else returns 0
		int isInvolutory(); //Returns 1 if the matrix is involutory else returns 0
		int isNilpotent(); //Returns 1 if the matrix is nilpotent else returns 0
		Matrix symmskew(); //expresses the matrix as a sum of a symmetric and skew symmetric matrix
		int duplicate(); //returns the number of duplicate numbers in the matrix
		Matrix additiveInv(); //finds the additive inverse of the matrix
};

Matrix::Matrix(int row,int column)
{
	r=row;
	c=column;
	for(int i=0;i<row;i++)
		for(int j=0;j<column;j++)
		{
			float x;
			cin>>x;
			m[i][j]=x/1.0;
		}
}

void Matrix::printMatrix()
{
	for(int i=0;i<r;i++)
		for(int j=0;j<c;j++)
			// Prints ' ' if j != n-1 else prints '\n'
          		cout << m[i][j] << " \n"[j == c-1];
}

Matrix Matrix::addition() {
    Matrix m1(r,c);
    Matrix m2(r,c);
    Matrix res(r,c);
    for(int i=0;i<r;i++) {
        for(int j=0;j<c;j++) {
            res.m[i][j] = m1.m[i][j] + m2.m[i][j];
        }
    }
   return res;
}

Matrix Matrix::subtraction() { //PR for subtraction
    Matrix m1(r,c);
    Matrix m2(r,c);
    Matrix res(r,c);
    for(int i=0;i<r;i++) {
        for(int j=0;j<c;j++) {
            res.m[i][j] = m1.m[i][j] - m2.m[i][j];
        }
    }
   return res;
}

Matrix Matrix::multiplication() {
    Matrix m1(r,c);
    Matrix m2(r,c);
    Matrix res(r,c);
    int i, j, k;
        for (i = 0; i < r; i++) {
            for (j = 0; j < r; j++) {
                res.m[i][j] = 0;
                for (k = 0; k < r; k++)
                    res.m[i][j] += m1.m[i][k]
                                 * m2.m[k][j];
            }
        }
   return res;
}

int Matrix::isIdentity() { //pr for isIdentity
    int flag=0;
    Matrix m1(r,c);
    for(int i=0;i<r;i++)
       {
           if(m1.m[i][i] == 1)
            flag ++;;
       }
       if(flag == m1.r)
       return 1;
       else
        return 0;
}

int Matrix::trace() {
    Matrix m1(r,c);
    int sum=0;
    for(int i=0;i<r;i++)
        sum  = sum + m1.m[i][i];
    return sum;
}

int Matrix::isSquare() {
    Matrix m1(r,c);
    if(r==c)
        return 1;
    else
        return 0;
}

int *Matrix::dimensions() {
    Matrix m1(r,c);
    int arr[2];
    arr[0]=r;
    arr[1]=c;
    return arr;

}

int *Matrix::gaussElimination()
{
	int x_arr[100];
	int *ptr=x_arr;
	for(int i=0;i<c;i++)
		for(int j=i+1;j<r;j++)
		{
			int x=m[j][i]/m[i][i];
			*ptr=x;
			ptr++;
			for(int k=0;k<c;k++)
				m[j][k]=m[j][k]-(x*m[i][k]);
			printMatrix();
			cout<<"\n";
		}
	return x_arr;
}

Matrix Matrix::columnSpace()
{
	gaussElimination();
	int k;
	for(int i=0;i<r;i++)
		if(m[i][i]==0)
		{
			k=i;
			break;
		}
	Matrix out(r,k);
	for(int i=0;i<r;i++)
		for(int j=0;j<k;j++)
		out.m[i][j]=m[i][j];
	return out;
}

Matrix Matrix::transpose()
{
	Matrix out(c,r);
	for(int i=0;i<r;i++)
		for(int j=0;j<c;j++)
			out.m[j][i]=m[i][j];
	return out;
}

double Matrix::determinant(int n,double mat[10][10])
{
    int k, subi, i, j, subj;
    double d=0;
    double submat[10][10];
    if (n == 2)
    {
        return( (mat[0][0] * mat[1][1]) - (mat[1][0] * mat[0][1]));
    }
    else
    {
        for(k = 0; k < n; k++)
        {
            subi = 0;
            for(i = 1; i < n; i++)
            {
                subj = 0;
                for(j = 0; j < n; j++)
                {
                    if (j == k)
                    {
                        continue;
                    }
                    submat[subi][subj] = mat[i][j];
                    subj++;
                }
                subi++;
            }
        d = d + (pow(-1 ,c) * mat[0][c] * determinant(r - 1 ,submat));
        }
    }
    return d;
}

Matrix *Matrix::LUdecomposition(){
	 Matrix upper(r,c);
         Matrix lower(r,c);
	 Matrix mat(r,c);
    // Decomposing matrix into Upper and Lower triangular matrix 
    for (int i = 0; i < r; i++) { 
        
	// Upper Triangular 
        for (int k = i; k < c; k++) { 
            //summation of lower[i][j]*upper[j][k]
            int sum = 0; 
            for (int j = 0; j < i; j++) 
                sum += (lower[i][j] * upper[j][k]); 
            upper[i][k] = mat[i][k] - sum; 
        } 
  
        // Lower Triangular 
        for (int k = i; k < c; k++) { 
            if (i == k) 
                lower[i][i] = 1;   // Diagonal as 1 
            else {
		//summation of lower[k][j]*upper[j][i]
                int sum = 0; 
                for (int j = 0; j < i; j++) 
                    sum += (lower[k][j] * upper[j][i]); 
		// Evaluating lower matrix
                lower[k][i] = (mat[k][i] - sum) / upper[i][i]; 
            } 
        } 
    }
	upper.printMatrix();
	return lower;
}


int Matrix::isIdempotent(){
	Matrix mat(r,c);
	Matrix result(r,c);
	int i, j, k,f=0;
        for (i = 0; i < r; i++) {
            for (j = 0; j < r; j++) {
                result.m[i][j] = 0;
                for (k = 0; k < r; k++)
                    result.m[i][j] += mat.m[i][k]
                                 * mat.m[k][j];
            }
        }
	for (i = 0; i < r; i++) {
            for (j = 0; j < r;) {
		    if(result.m[i][j] == mat.m[i][j])
			    j++;
		    else{
			    f = 1;
			    break;
		    }
	    }
	}
	if(f==0)
		return 1;  //indicates that the matrix is idempotent
	else
		return 0;  //indicates that the matrix is not idempotent
}

int Matrix::isInvolutory(){
	Matrix mat(r,c);
	Matrix result(r,c);
	int i, j, k,f=0;
        for (i = 0; i < r; i++) {
            for (j = 0; j < r; j++) {
                result.m[i][j] = 0;
                for (k = 0; k < r; k++)
                    result.m[i][j] += mat.m[i][k]
                                 * mat.m[k][j];
            }
        }
	for (i = 0; i < r; i++) {
            for (j = 0; j < r;j++) {
		    if(i == j){
			    if(result.m[i][j] == 1)
				    continue;
			    else{
				    f=1;
				    break;
			    }
		    }
		    elseif(result.m[i][j]==0)
		            continue;
	            else{
			    f = 1;
			    break;
		    }
	    }
	}
	if(f==0)
		return 1;  //indicates that the matrix is involutory
	else
		return 0;  //indicates that the matrix is not involutory
}


Matrix Matrix::additiveInv() {
    Matrix m1(c,r);
    Matrix out(c,r);
    for(int i=0;i<m1.r;i++)
    {
        for(int j=0;j<m1.c;j++)
        {
            out.m[i][j] = (-1)*m1.m[i][j];
        }
    }
    return out;
}

#<<<<<<< main
#int Matrix::isInvertible()
#{
#    Matrix m2(r,c);
#    float m3[10][10];
#    m3[10][10]= m2.m[10][10];
#    float d1=determinant(r,m3);
#    if(isSquare() == 1)
#    {
#        if(d1 != 0)
#            return 1;
#    }
#    else
#        return 0;
#}


#=======
#>>>>>>> main
int main()
{
	Matrix m(4,4);
	cout<<"\n";
	m.printMatrix();
	cout<<"\n";

    Matrix resu = m.addition();
    resu.printMatrix();
    cout<<"\n";

    resu = m.subtraction();
    resu.printMatrix();
    cout<<"\n";


	Matrix out=m.transpose();
	out.printMatrix();
	cout<<"\n";

    cout<<"gauss";
	int *y=m.gaussElimination();
	m.printMatrix();
	cout<<"\n";

	Matrix out1=m.columnSpace();
	out1.printMatrix();

	return 0;
}
