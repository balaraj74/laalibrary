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
		Matrix(int row, int column, bool fill=true);
		/*
		 * boolean flag fill defaults to true to prevent this change from
		 * being a breaking change. If you do not want to be asked to input a
		 * value for every single element in the matrix, set fill to false.
		 */

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
		double determinant(int , float [100][100]);
		Matrix columnSpace();
		Matrix transpose();
		Matrix SilentTranspose();
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
		int duplicate(); //returns the number of duplicate numbers in the matrix
		Matrix additiveInv(); //finds the additive inverse of the matrix
		Matrix cofactor(float [100][100],int ,int ,int );

		// since this function computes two matrices, we return a pointer to a matrix array
		// TODO check memory safety of this
		Matrix* symmskew(); //expresses the matrix as a sum of a symmetric and skew symmetric matrix

		// operator overloading for addition and subtraction of like matrices
		Matrix operator + (const Matrix& other)
		{
			if (r == other.r && c == other.c)
			{
				// we can perform operations on them
				Matrix result(r, c, false);

				for (int i = 0; i < r; i++)
				{
					for (int j = 0; j < c; j++)
					{
						result.m[i][j] = m[i][j] + other.m[i][j];
					}
				}
				cout << "Added Matrices successfully." << endl;
				return result;
			}
			else
			{
				cout << "Cannot add two unlike matrices" << endl;
				return Matrix(1,1, false);
			}
		};

		Matrix operator - (const Matrix& other)
		{
			if (r == other.r && c == other.c)
			{
				Matrix result(r, c, false);

				for (int i = 0; i < r; i++)
				{
					for (int j = 0; j < c; j++)
					{
						result.m[i][j] = m[i][j] - other.m[i][j];
					}
				}
				cout << "Subtracted Matrices successfully." << endl;
				return result;
			}
			else
			{
				cout << "Cannot subtract two unlike matrices" << endl;
				return Matrix(1,1, false);
			}
		};

		// overload scalar multiplication
		Matrix operator * (const int& scalar)
		{
			Matrix result(r, c, false);

			for (int i = 0; i < r; i++)
			{
				for (int j = 0; j < c; j++)
				{
					result.m[i][j] = m[i][j] * scalar;
				}
			}
			cout << "Successfully multiplied scalar" << endl;
			return result;
		};

		// overload scalar division
		Matrix operator / (const int& scalar)
		{
			Matrix result(r, c, false);

			for (int i = 0; i < r; i++)
			{
				for (int j = 0; j < c; j++)
				{
					result.m[i][j] = m[i][j] / scalar;
				}
			}
			cout << "Subtracted Matrices successfully." << endl;
			return result;
		};
};

Matrix::Matrix(int row, int column, bool fill)
{
	r=row;
	c=column;
	if (fill == true)
	{
		for(int i=0;i<row;i++)
			for(int j=0;j<column;j++)
			{
				float x;
				cin>>x;
				m[i][j]=x/1.0;
			}
	}
	else
	{
		for(int i=0;i<row;i++)
		{
			for(int j=0;j<column;j++)
			{
				m[i][j]=0;
			}
		}
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

int Matrix::isIdentity() { //pr for isidentity
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
			float x=m[j][i]/m[i][i];
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

Matrix Matrix::SilentTranspose()
{
	Matrix out(c, r, false);
	for(int i=0;i<r;i++)
		for(int j=0;j<c;j++)
			out.m[j][i]=m[i][j];
	return out;
}

double Matrix::determinant(int n,float mat[100][100])
{
    double D = 0;

    //  Base case : if matrix contains single element
    if (n == 1)
        return mat[0][0];

     Matrix temp(10,10); // To store cofactors

    int sign = 1;  // To store sign multiplier


    for (int f = 0; f < n; f++)
    {
        // Getting Cofactor of A[0][f]
        temp = cofactor(mat, 0, f, n);
        D = D + sign * mat[0][f] * determinant(n-1,temp.m);

        // terms are to be added with alternate sign
        sign = -sign;
    }

    return D;
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
                sum += (lower.m[i][j] * upper.m[j][k]);
            upper.m[i][k] = mat.m[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < c; k++) {
            if (i == k)
                lower.m[i][i] = 1;   // Diagonal as 1
            else {
		//summation of lower[k][j]*upper[j][i]
                int sum = 0;
                for (int j = 0; j < i; j++)
                    sum += (lower.m[k][j] * upper.m[j][i]);
		// Evaluating lower matrix
                lower.m[k][i] = (mat.m[k][i] - sum) / upper.m[i][i];
            }
        }
    }
	upper.printMatrix();
	lower.printMatrix();
	return 0;
}

Matrix *Matrix::SVD(){
	int minDim = (r < c) ? r : c;
	int maxDim = (r > c) ? r : c;
	
	Matrix *result = new Matrix[3]{Matrix(r, minDim, false), Matrix(minDim, c, false), Matrix(c, c, false)};
	Matrix &U = result[0];
	Matrix &S = result[1];
	Matrix &V = result[2];
	
	Matrix AtA(c, c, false);
	
	for(int i=0; i<c; i++){
		for(int j=0; j<c; j++){
			AtA.m[i][j] = 0;
			for(int k=0; k<r; k++){
				AtA.m[i][j] += m[k][i] * m[k][j];
			}
		}
	}
	
	for(int i=0; i<c; i++){
		for(int j=0; j<c; j++){
			V.m[i][j] = (i==j) ? 1.0 : 0.0;
		}
	}
	
	int maxIter = 100;
	float tolerance = 1e-10;
	
	for(int iter=0; iter<maxIter; iter++){
		float maxVal = 0;
		int p=0, q=1;
		
		for(int i=0; i<c; i++){
			for(int j=i+1; j<c; j++){
				if(fabs(AtA.m[i][j]) > maxVal){
					maxVal = fabs(AtA.m[i][j]);
					p = i;
					q = j;
				}
			}
		}
		
		if(maxVal < tolerance) break;
		
		float theta;
		if(fabs(AtA.m[p][p] - AtA.m[q][q]) < tolerance){
			theta = M_PI / 4.0;
		} else {
			theta = 0.5 * atan(2.0 * AtA.m[p][q] / (AtA.m[p][p] - AtA.m[q][q]));
		}
		
		float c_val = cos(theta);
		float s_val = sin(theta);
		
		float temp[100][100];
		for(int i=0; i<c; i++){
			for(int j=0; j<c; j++){
				temp[i][j] = AtA.m[i][j];
			}
		}
		
		for(int i=0; i<c; i++){
			if(i != p && i != q){
				AtA.m[p][i] = c_val * temp[p][i] + s_val * temp[q][i];
				AtA.m[i][p] = AtA.m[p][i];
				AtA.m[q][i] = -s_val * temp[p][i] + c_val * temp[q][i];
				AtA.m[i][q] = AtA.m[q][i];
			}
		}
		
		AtA.m[p][p] = c_val * c_val * temp[p][p] + 2.0 * c_val * s_val * temp[p][q] + s_val * s_val * temp[q][q];
		AtA.m[q][q] = s_val * s_val * temp[p][p] - 2.0 * c_val * s_val * temp[p][q] + c_val * c_val * temp[q][q];
		AtA.m[p][q] = 0;
		AtA.m[q][p] = 0;
		
		float tempV[100][100];
		for(int i=0; i<c; i++){
			for(int j=0; j<c; j++){
				tempV[i][j] = V.m[i][j];
			}
		}
		
		for(int i=0; i<c; i++){
			V.m[i][p] = c_val * tempV[i][p] + s_val * tempV[i][q];
			V.m[i][q] = -s_val * tempV[i][p] + c_val * tempV[i][q];
		}
	}
	
	float eigenvalues[100];
	for(int i=0; i<c; i++){
		eigenvalues[i] = AtA.m[i][i];
	}
	
	int indices[100];
	for(int i=0; i<c; i++) indices[i] = i;
	
	for(int i=0; i<c-1; i++){
		for(int j=i+1; j<c; j++){
			if(eigenvalues[indices[i]] < eigenvalues[indices[j]]){
				int temp = indices[i];
				indices[i] = indices[j];
				indices[j] = temp;
			}
		}
	}
	
	Matrix Vsorted(c, c, false);
	float sortedEigenvalues[100];
	for(int i=0; i<c; i++){
		sortedEigenvalues[i] = eigenvalues[indices[i]];
		for(int j=0; j<c; j++){
			Vsorted.m[j][i] = V.m[j][indices[i]];
		}
	}
	
	for(int i=0; i<minDim; i++){
		for(int j=0; j<c; j++){
			S.m[i][j] = 0;
		}
		if(i < c){
			S.m[i][i] = sqrt(fabs(sortedEigenvalues[i]));
		}
	}
	
	for(int i=0; i<minDim; i++){
		float sigma = S.m[i][i];
		if(sigma > tolerance){
			for(int j=0; j<r; j++){
				U.m[j][i] = 0;
				for(int k=0; k<c; k++){
					U.m[j][i] += m[j][k] * Vsorted.m[k][i];
				}
				U.m[j][i] /= sigma;
			}
		} else {
			for(int j=0; j<r; j++){
				U.m[j][i] = (i < r && j == i) ? 1.0 : 0.0;
			}
		}
	}
	
	cout<<"U Matrix ("<<r<<"x"<<minDim<<"):"<<endl;
	U.printMatrix();
	cout<<endl;
	
	cout<<"Sigma Matrix ("<<minDim<<"x"<<c<<"):"<<endl;
	S.printMatrix();
	cout<<endl;
	
	cout<<"V Transpose Matrix ("<<c<<"x"<<c<<"):"<<endl;
	Matrix Vt(c, c, false);
	for(int i=0; i<c; i++){
		for(int j=0; j<c; j++){
			Vt.m[i][j] = Vsorted.m[j][i];
		}
	}
	Vt.printMatrix();
	cout<<endl;
	
	return result;
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
		    else if(result.m[i][j]==0)
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

Matrix* Matrix::symmskew()
{
	if (r == c)
	{
		// get copies of both matrices needed
		Matrix transpose = SilentTranspose();
		Matrix current = (*this);

		// make empty placeholder matrices
		Matrix symmetric(r, c, false);
		Matrix skew_symmetric(r, c, false);

		// perform operations
		symmetric = (current + transpose)/2;
		skew_symmetric = (current - transpose)/2;

		// initialize pointer to array of matrices
		Matrix* result_pointer = NULL;

		// assign positions
		result_pointer[0] = symmetric;
		result_pointer[1] = skew_symmetric;

		// return
		return result_pointer;
	}
	else
	{
		cout << "Matrix must be square \n";
		return NULL;
	}
}


//takes in matrix , row no,column no, and size of matrix
Matrix Matrix::cofactor(float a[100][100],int rw,int cl,int N)
{
    int i = 0, j = 0;
    Matrix temp(c,r);

    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            //  Copying into temporary matrix only those element
            //  which are not in given row and column
            if (row != rw && col != cl)
            {
                temp.m[i][j++] = a[row][col];


                if (j == N - 1)
                {
                    j = 0;
                    i++;
                }
            }
        }
    }
    return temp;
}

/*
int Matrix::isInvertible()
{
    Matrix m2(r,c);
    float m3[10][10];
    m3[10][10]= m2.m[10][10];
    float d1=determinant(r,m3);
    if(isSquare() == 1)
    {
        if(d1 != 0)
            return 1;
    }
    else
        return 0;
}
*/


// driver code
int main()
{
	cout<<"Enter matrix dimensions (rows cols):"<<endl;
	int rows, cols;
	cin >> rows >> cols;
	
	cout<<"Enter matrix elements:"<<endl;
	Matrix m(rows, cols);
	cout<<"\nOriginal Matrix:"<<endl;
	m.printMatrix();
	cout<<"\n";

	cout<<"Performing SVD..."<<endl;
	Matrix *svd_result = m.SVD();
	
	return 0;
}

