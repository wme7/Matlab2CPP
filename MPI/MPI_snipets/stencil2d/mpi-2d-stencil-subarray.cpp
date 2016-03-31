// author: Ugo Varetto
// data exchange between 2D arrays in a 2D MPI cartesian grid 
#include "stencil2D.h"

typedef double REAL;


//------------------------------------------------------------------------------
// Specialization for array elment type
template <> MPI_Datatype CreateArrayElementType< double >() { return MPI_DOUBLE_PRECISION; }

//------------------------------------------------------------------------------
template < typename T >
void InitArray( T* pdata, const Array2D& g, const T& value ) {
    Array2DAccessor< T > a( pdata, g );
    for( int row = 0; row != a.Layout().height; ++row ) {
        for( int column = 0; column != a.Layout().width; ++column ) {
            a( column, row ) = value;

        }
    }
}


//------------------------------------------------------------------------------
template < typename T >
void Compute( T* pdata, const Array2D& g ) {}

//------------------------------------------------------------------------------
template < typename T >
bool TerminateCondition( T* pdata, const Array2D& g ) { return true; }


//------------------------------------------------------------------------------
int main( int argc, char** argv ) {
    //TestSubRegionExtraction();
    int numtasks = 0; 
    // Init, world size     
    MPI_Init( &argc, &argv );
    MPI_Errhandler_set( MPI_COMM_WORLD, MPI_ERRORS_RETURN );
    MPI_Comm_size( MPI_COMM_WORLD, &numtasks );
    // 2D square MPI cartesian grid 
    const int DIM = int( std::sqrt( double( numtasks ) ) );
    if( DIM * DIM != numtasks ) {
        std::cerr << "Numer of MPI tasks must be a perfect square" << std::endl;
        return 1; 
    }
    std::vector< int > dims( 2, DIM );
    std::vector< int > periods( 2, 1 ); //periodic in both dimensions
    const int reorder = 0; //false - no reorder, is it actually used ?
    MPI_Comm cartcomm; // communicator for cartesian grid
    MPI_Cart_create( MPI_COMM_WORLD, 2, &dims[ 0 ], &periods[ 0 ], reorder, &cartcomm ); 
    // current mpi task id of this process
    int task = -1;
    MPI_Comm_rank( cartcomm, &task );
    
    std::vector< int > coords( 2, -1 );
    MPI_Cart_coords( cartcomm, task, 2, &coords[ 0 ] );
      
    std::ostringstream ss;
    ss << coords[ 0 ] << '_' << coords[ 1 ];
    std::ofstream os( ss.str().c_str() );
    os << "Rank:  " << task << std::endl
       << "Coord: " << coords[ 0 ] << ", " << coords[ 1 ] << std::endl;
    
    os << std::endl << "Compute grid" << std::endl;
    PrintCartesianGrid( os, cartcomm, DIM, DIM );
    os << std::endl;
    
    // Init data
    int localWidth = 16;
    int localHeight = 16;
    int stencilWidth = 5;
    int stencilHeight = 5;
    int localTotalWidth = localWidth + 2 * ( stencilWidth / 2 );
    int localTotalHeight = localHeight + 2 * ( stencilHeight / 2 );
    std::vector< REAL > dataBuffer( localTotalWidth * localTotalHeight, -1 );  
    Array2D localArray( localTotalWidth, localTotalHeight, localTotalWidth );
    // Create transfer info arrays
    typedef std::vector< TransferInfo > VTI;
    std::pair< VTI, VTI > transferInfoArrays =
        CreateSendRecvArrays( &dataBuffer[ 0 ], cartcomm, task, localArray, stencilWidth, stencilHeight );     
    Array2D core = SubArrayRegion( localArray, stencilWidth, stencilHeight, CENTER );
    InitArray( &dataBuffer[ 0 ], core, REAL( task ) ); //init with this MPI task id
    os << localWidth << " x " << localHeight << " grid size" << std::endl; 
    os << localTotalWidth << " x " << localTotalHeight << " total(with ghost/halo regions) grid size" << std::endl; 
    os << stencilWidth << " x " << stencilHeight << " stencil\n" << std::endl;
    os << "Array" << std::endl;
    Print( &dataBuffer[ 0 ], localArray, os );
    os << std::endl;
    // Exchange data and compute until condition met
    do {
        ExchangeData( transferInfoArrays.first, transferInfoArrays.second );
        Compute( &dataBuffer[ 0 ], core );
    } while( !TerminateCondition( &dataBuffer[ 0 ], core ) );
    os << "Array after exchange" << std::endl;    
    MPI_Finalize();
    Print( &dataBuffer[ 0 ], localArray, os );   
    return 0;
}
