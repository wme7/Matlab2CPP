#pragma once

// author: Ugo Varetto
//
// utility functions for 2D stencil computations on regular grids with 
// direct GPU-GPU data exchange
//
// intended for use with mvapich2 version >= 1.8RC1

#include <mpi.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <unistd.h> 
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>

#ifdef __CUDACC__
#define ACC_ __host__ __device__
#else
#define ACC_
#endif


//------------------------------------------------------------------------------
// 2D layout information
struct Array2D {
    int width;
    int height;
    int xOffset;
    int yOffset;
    int rowStride; // currently not used
    ACC_ Array2D( int w, int h, int rs, int xoff = 0, int yoff = 0 ) :
        width( w ), height( h ), xOffset( xoff ), yOffset( yoff ),
        rowStride( rs )
    {}
    ACC_ Array2D() : width( 0 ), height( 0 ), xOffset( 0 ), yOffset( 0 ),rowStride( 0 )
    {}
};

std::ostream& operator<<( std::ostream& os, const Array2D& a ) {
    os << "width:  " << a.width << ", " 
       << "height: " << a.height << ", "
       << "x offset: " << a.xOffset << ", "
       << "y offset: " << a.yOffset;
    return os;
}

//------------------------------------------------------------------------------
// Accessor for 2D arrays: Allow for random access to 2D domain given pointer 
// and layout information 
template < typename T >
class Array2DAccessor {
public:
    ACC_ const T& operator()( int x, int y ) const { 
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                    layout_.rowStride * y + x ); 
    }
    ACC_ T& operator()( int x, int y ) {
        
        return *( data_ + 
                  ( layout_.yOffset * layout_.rowStride + layout_.xOffset ) + //constant
                  layout_.rowStride * y + x ); 
    }
    ACC_ Array2DAccessor() : data_( 0 ) {}
    ACC_ Array2DAccessor( T* data, const Array2D& layout ) : data_( data ), layout_( layout) {}
    ACC_ const Array2D& Layout() const { return layout_; }
private:
    Array2D layout_;
    T* data_;
};

//------------------------------------------------------------------------------
// Region ids: Used to identify areas in the local grid
enum RegionID { TOP_LEFT,    TOP_CENTER,    TOP_RIGHT,
                CENTER_LEFT, CENTER,        CENTER_RIGHT,
                BOTTOM_LEFT, BOTTOM_CENTER, BOTTOM_RIGHT,
                TOP, LEFT, BOTTOM, RIGHT };

//------------------------------------------------------------------------------
// MPI grid cell  ids: Used to identify neighbors in the global MPI cartesian grid
enum MPIGridCellID { MPI_TOP_LEFT,    MPI_TOP_CENTER,    MPI_TOP_RIGHT,
                     MPI_CENTER_LEFT,                    MPI_CENTER_RIGHT,
                     MPI_BOTTOM_LEFT, MPI_BOTTOM_CENTER, MPI_BOTTOM_RIGHT };

//------------------------------------------------------------------------------
// Print content of 2D array given pointer to data and layout info
template < typename T >
void Print( T* pdata, const Array2D& g, std::ostream& os ) {
    Array2DAccessor< T > a( pdata, g );
    for( int row = 0; row != a.Layout().height; ++row ) {
        for( int column = 0; column != a.Layout().width; ++column ) {
            os << a( column, row ) << ' ';

        }
        os << std::endl;
    }
}

//------------------------------------------------------------------------------
//Compute Array2D layout from region id of local grid
//Possible to use templated function specialized with RegionID
inline Array2D SubArrayRegion( const Array2D& g, 
                               int stencilWidth, 
                               int stencilHeight,
                               RegionID rid ) {
    int width = 0;
    int height = 0;
    int xoff = 0;
    int yoff = 0;
    const int stride = g.width;
    const int ghostRegionWidth  = stencilWidth  / 2;
    const int ghostRegionHeight = stencilHeight / 2;
    switch( rid ) {
        case TOP_LEFT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break;
        case TOP_CENTER:
            width = g.width - 2 * ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case TOP_RIGHT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case CENTER_LEFT:
            width = ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset; 
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case CENTER: //core space
            width = g.width - 2 * ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case CENTER_RIGHT:
            width = ghostRegionWidth;
            height = g.height - 2 * ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset + ghostRegionHeight;
            break;
        case BOTTOM_LEFT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case BOTTOM_CENTER:
            width = g.width - 2 * ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + ghostRegionWidth;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case BOTTOM_RIGHT:
            width = ghostRegionWidth;
            height = ghostRegionHeight;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case TOP:
            width = g.width;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break;
        case RIGHT:
            width = ghostRegionWidth;
            height = g.height;
            xoff = g.xOffset + g.width - ghostRegionWidth;
            yoff = g.yOffset;
            break;
        case BOTTOM:
            width = g.width;
            height = ghostRegionHeight;
            xoff = g.xOffset;
            yoff = g.yOffset + g.height - ghostRegionHeight;
            break;
        case LEFT:
            width = ghostRegionWidth;
            height = g.height;
            xoff = g.xOffset;
            yoff = g.yOffset;
            break; 
        default:
            break; 
    }   
    return Array2D( width, height, stride, xoff, yoff );
}

//------------------------------------------------------------------------------
// Point of customization for client code: specialize this function for the
// array element data type of choice
template < typename T > MPI_Datatype CreateArrayElementType();


//------------------------------------------------------------------------------
template < typename T >
MPI_Datatype CreateMPISubArrayType( const Array2D& g, const Array2D& subgrid ) {
    int dimensions = 2;
    int sizes[] = { g.width, g.height };
    int subsizes[] = { subgrid.width, subgrid.height };
    int offsets[] = { subgrid.xOffset, subgrid.yOffset };
    int order = MPI_ORDER_C;
    MPI_Datatype arrayElementType = CreateArrayElementType< T >();// array element type
    MPI_Datatype newtype;
    MPI_Type_create_subarray( dimensions,
                              sizes,
                              subsizes,
                              offsets,
                              order,
                              arrayElementType,
                              &newtype );
    MPI_Type_commit( &newtype );
    return newtype;
}

//------------------------------------------------------------------------------
// Return MPI task id from offset in MPI cartesian grid
inline int OffsetTaskId( MPI_Comm comm, int xOffset, int yOffset ) {
    int thisRank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &thisRank );
    int coord[] = { -1, -1 }; 
    MPI_Cart_coords( comm, thisRank, 2, coord );
//    printf( "%d %d\n", coord[ 0 ], coord[ 1 ] );
    coord[ 0 ] += xOffset;
    coord[ 1 ] += yOffset;
    int rank = -1;
    MPI_Cart_rank( comm, coord, &rank );
//    printf( "In rank: %d, offset: %d, %d; out rank: %d\n", thisRank, xOffset, yOffset, rank ); 
    return rank; 
}


//------------------------------------------------------------------------------
// Offset from current MPI taks in MPI cartesian grid
struct Offset {
    int x;
    int y;
    Offset( int ox, int oy ) : x( ox ), y( oy ) {}
};

// compute the offset in the MPI grid from the Region id
// of the local grid;
// assuming a row-major top-to-bottom MPI compute grid
// i.e. topmost row is 0 
inline Offset MPIOffsetRegion( MPIGridCellID mpicid ) {
    int xoff = 0;
    int yoff = 0;
    switch( mpicid ) {
    case MPI_TOP_LEFT:
        xoff =  -1;
        yoff =  -1;
        break;
    case MPI_TOP_CENTER:
        xoff =  0;
        yoff = -1;
        break;
    case MPI_TOP_RIGHT:
        xoff =  1;
        yoff = -1;
        break;
    case MPI_CENTER_LEFT:
        xoff = -1;
        yoff =  0;
        break;
    case MPI_CENTER_RIGHT:
        xoff = 1;
        yoff = 0;
        break;
    case MPI_BOTTOM_LEFT:
        xoff = -1;
        yoff =  1;
        break;
    case MPI_BOTTOM_CENTER:
        xoff =  0;
        yoff =  1;
        break;
    case MPI_BOTTOM_RIGHT:
        xoff =  1;
        yoff =  1;
        break;
    default:
        break;
    }
    return Offset( xoff, yoff );
}

//------------------------------------------------------------------------------
// Danta transfer information to be used by MPI send/receive operations
struct TransferInfo {
    int srcTaskId;
    int destTaskId;
    int tag;
    void* data;
    MPI_Request request; // currently not used
    MPI_Datatype type;
    MPI_Comm comm;
};

//------------------------------------------------------------------------------
// Create entry with information for data transfer from a remote MPI task
// to a local subregion of a 2d array
// IN: remote MPI task, local target memory region, data pointer, sub-array layout
// OUT: transfer information including endpoints and MPI datatype matching the
//      2d array layout passed as input 
template < typename T > 
TransferInfo CreateReceiveInfo( T* pdata, MPI_Comm cartcomm, int rank, 
			        MPIGridCellID remoteSource, RegionID localTargetRegion,
                                Array2D& g, int stencilWidth, int stencilHeight, int tag ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.destTaskId = rank;
    ti.tag = tag;
    ti.type = CreateMPISubArrayType< T >( g, 
                                          SubArrayRegion( g, stencilWidth, stencilHeight, localTargetRegion ) );
    Offset offset = MPIOffsetRegion( remoteSource ); 
    ti.srcTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );

// printf( "source %d dest %d\n", ti.srcTaskId, ti.destTaskId ); 
  
    return ti;     
}
 
//------------------------------------------------------------------------------
// Create entry with information for data transfer from a local memory area
// inside the core space(local grid minus ghost regions) to a remote MPI task
// IN: local region, remote MPI task, data pointer, sub-array layout
// OUT: transfer information including endpoints and MPI datatype matching the
//      2d array layout passed as input 
template< typename T > 
TransferInfo CreateSendInfo( T* pdata, MPI_Comm cartcomm, int rank, 
                             MPIGridCellID remoteTarget, RegionID localSourceRegion, Array2D& g,
                             int stencilWidth, int stencilHeight, int tag ) {
    TransferInfo ti;
    ti.comm = cartcomm;
    ti.data = pdata;
    ti.srcTaskId = rank;
    ti.tag = tag;
    Array2D core = SubArrayRegion( g, stencilWidth, stencilHeight, CENTER );
    ti.type = CreateMPISubArrayType< T >( g, 
                                          SubArrayRegion( core, stencilWidth, stencilHeight, localSourceRegion ) );
    Offset offset = MPIOffsetRegion( remoteTarget ); 
    ti.destTaskId = OffsetTaskId( cartcomm, offset.x, offset.y );  
    return ti;     
}

//------------------------------------------------------------------------------
// Iterate over arrays of data transfer info and perform actual MPI data transfers
inline void ExchangeData( std::vector< TransferInfo >& recvArray,
                          std::vector< TransferInfo >& sendArray ) {

    std::vector< MPI_Request  > requests( recvArray.size() + sendArray.size() );
    for( int i = 0; i != recvArray.size(); ++i ) {
        TransferInfo& t = recvArray[ i ];
        MPI_Irecv( t.data, 1, t.type, t.srcTaskId, t.tag, t.comm, &( requests[ i ] ) );  
    }
    for( int i = 0; i != sendArray.size(); ++i ) {
        TransferInfo& t = sendArray[ i ];
        MPI_Isend( t.data, 1, t.type, t.destTaskId, t.tag, t.comm, &( requests[ recvArray.size() + i ] ) );  
    }
    std::vector< MPI_Status > status( recvArray.size() + sendArray.size() );
    MPI_Waitall( requests.size(), &requests[ 0 ], &status[ 0 ] );  
}

//------------------------------------------------------------------------------
// Create pair of <recv, send> info array 
template < typename T > 
std::pair< std::vector< TransferInfo >,
           std::vector< TransferInfo > > 
CreateSendRecvArrays( T* pdata, MPI_Comm cartcomm, int rank, Array2D& g,
                      int stencilWidth, int stencilHeight ) {
    std::vector< TransferInfo > ra;
    std::vector< TransferInfo > sa;
    // send regions: data extracted from core(CENTER) region
    RegionID localSendSource[] = { TOP_LEFT,    TOP,    TOP_RIGHT,
                                    LEFT,                RIGHT,
                                    BOTTOM_LEFT, BOTTOM, BOTTOM_RIGHT };
    // recv regions: data inserted into local grid
    RegionID localRecvTarget[]  = { BOTTOM_RIGHT,  BOTTOM_CENTER,    BOTTOM_LEFT,
                                    CENTER_RIGHT,                CENTER_LEFT,
                                    TOP_RIGHT, TOP_CENTER, TOP_LEFT };

    // remote send targets
    MPIGridCellID remoteSendTarget[] = { MPI_TOP_LEFT,    MPI_TOP_CENTER,    MPI_TOP_RIGHT,
                                         MPI_CENTER_LEFT,                    MPI_CENTER_RIGHT,
                                         MPI_BOTTOM_LEFT, MPI_BOTTOM_CENTER, MPI_BOTTOM_RIGHT };


    // remote recv sources
    MPIGridCellID remoteRecvSource[] = { MPI_BOTTOM_RIGHT,  MPI_BOTTOM_CENTER, MPI_BOTTOM_LEFT,
                                         MPI_CENTER_RIGHT,                     MPI_CENTER_LEFT,
                                         MPI_TOP_RIGHT,     MPI_TOP_CENTER,    MPI_TOP_LEFT };

    RegionID*      lss = &localSendSource [ 0 ];
    MPIGridCellID* rst = &remoteSendTarget[ 0 ];
    RegionID*      lrt = &localRecvTarget [ 0 ];
    MPIGridCellID* rrs = &remoteRecvSource[ 0 ];
    const size_t sz = sizeof( localSendSource ) / sizeof( RegionID );
    const RegionID* end = lss + sz;
    while( lss != end ) {
        ra.push_back( CreateReceiveInfo( pdata,    // data 
                                         cartcomm, // MPI cartesian communicator 
                                         rank,     // rank of this process
                                         *rrs,     // MPI cell id of process to read data from
                                         *lrt,     // local receive target i.e. id of local area to fill
                                            g,     // local 2D Array
                                         stencilWidth, stencilHeight,
                                         *lss ) );  // tag
        sa.push_back( CreateSendInfo( pdata, cartcomm, rank,
                                      *rst, // MPI cell id of process to send data to
                                      *lss, // local send source i.e. id of local area to extract data from
                                         g, // local 2D Array
                                      stencilWidth, stencilHeight,
                                      *lss ) ); // tag  

        ++lss;
        ++rst;
        ++lrt;
        ++rrs;                                           
    }

    return std::make_pair( ra, sa ); 
}


//------------------------------------------------------------------------------
inline void TestSubRegionExtraction() {
    const int w = 32;
    const int h = 32;
    const int stencilWidth = 5;
    const int stencilHeight = 5;
    const int totalWidth = w + stencilWidth / 2;
    const int totalHeight = h + stencilHeight / 2;
    // Grid(core + halo)
    std::vector< int > data( totalWidth * totalHeight, 0 );
    Array2D grid( totalWidth, totalHeight, totalWidth );
    Array2D topleft = SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_LEFT );
    Array2D topcenter = SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_CENTER );
    Array2D topright= SubArrayRegion( grid, stencilWidth, stencilHeight, TOP_RIGHT );
    Array2D centerleft = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER_LEFT );
    Array2D center = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER );
    Array2D centerright = SubArrayRegion( grid, stencilWidth, stencilHeight, CENTER_RIGHT );
    Array2D bottomleft = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_LEFT );
    Array2D bottomcenter = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_CENTER );
    Array2D bottomright = SubArrayRegion( grid, stencilWidth, stencilHeight, BOTTOM_RIGHT );
  
    std::cout << "\nGRID TEST\n";
    
    std::cout << "Width: " << totalWidth << ", " << "Height: " << totalHeight << std::endl;
    std::cout << "Stencil: " << stencilWidth << ", " << stencilHeight << std::endl;
 
    std::cout << "top left:      " << topleft      << std::endl;
    std::cout << "top center:    " << topcenter    << std::endl;
    std::cout << "top right:     " << topright     << std::endl;
    std::cout << "center left:   " << centerleft   << std::endl;
    std::cout << "center:        " << center       << std::endl;
    std::cout << "center right:  " << centerright  << std::endl;
    std::cout << "bottom left:   " << bottomleft   << std::endl;
    std::cout << "bottom center: " << bottomcenter << std::endl;
    std::cout << "bottom right:  " << bottomright  << std::endl;

    std::cout << "\nSUBGRID TEST\n";

    // Core space(area of stencil application)
    Array2D core = center;
    topleft = SubArrayRegion( core, stencilWidth, stencilHeight, TOP_LEFT );
    topcenter = SubArrayRegion( core, stencilWidth, stencilHeight, TOP_CENTER );
    topright= SubArrayRegion( core, stencilWidth, stencilHeight, TOP_RIGHT );
    centerleft = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER_LEFT );
    center = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER );
    centerright = SubArrayRegion( core, stencilWidth, stencilHeight, CENTER_RIGHT );
    bottomleft = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_LEFT );
    bottomcenter = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_CENTER );
    bottomright = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM_RIGHT );
    Array2D top = SubArrayRegion( core, stencilWidth, stencilHeight, TOP );
    Array2D right = SubArrayRegion( core, stencilWidth, stencilHeight, RIGHT );
    Array2D bottom = SubArrayRegion( core, stencilWidth, stencilHeight, BOTTOM );
    Array2D left = SubArrayRegion( core, stencilWidth, stencilHeight, LEFT );

    std::cout << "Width: " << core.width << ", " << "Height: " << core.height << std::endl;
    std::cout << "Stencil: " << stencilWidth << ", " << stencilHeight << std::endl;
    
    std::cout << "top left:      " << topleft      << std::endl;
    std::cout << "top center:    " << topcenter    << std::endl;
    std::cout << "top right:     " << topright     << std::endl;
    std::cout << "center left:   " << centerleft   << std::endl;
    std::cout << "center:        " << center       << std::endl;
    std::cout << "center right:  " << centerright  << std::endl;
    std::cout << "bottom left:   " << bottomleft   << std::endl;
    std::cout << "bottom center: " << bottomcenter << std::endl;
    std::cout << "bottom right:  " << bottomright  << std::endl;
    std::cout << "top:           " << top          << std::endl;
    std::cout << "right:         " << right        << std::endl;
    std::cout << "bottom:        " << bottom       << std::endl;
    std::cout << "left:          " << left         << std::endl;
}

//------------------------------------------------------------------------------
inline void PrintCartesianGrid( std::ostream& os, MPI_Comm comm, int rows, int columns ) {

    std::vector< std::vector< int > > grid( rows, std::vector< int >( columns, -1 ) );

    for( int r = 0; r != rows; ++r ) {
        for( int c = 0; c != columns; ++c ) {
            int coords[] = { -1, -1 };
            MPI_Cart_coords( comm, r * columns + c, 2, coords );
            grid[ coords[ 0 ] ][ coords[ 1 ] ] = r * columns + c;   
        }        
    }
    for( int r = 0; r != rows; ++r ) {
        for( int c = 0; c != columns; ++c ) {
            os << grid[ r ][ c ] << ' '; 
        }
        os << std::endl;
    }   
}
