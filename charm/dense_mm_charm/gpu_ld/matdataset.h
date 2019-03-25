#ifndef _MATDATASET_H_
#define _MATDATASET_H_

#include <fstream>
#ifndef _DOUBLE_PRECISION
#define PRECISION float
#else
#define PRECISION double
#endif

#define DS_MAGIC      0x1e4560f2
#define DS_VERSION    0x0d7f5092

using namespace std;

class MatrixDataset {
public:
    typedef struct {
      PRECISION * elements;
      int         width;
      int         height;
    } Matrix;

    typedef struct {
        int magic;
        int version;
        int numelements;
        int width;
        int height;
    } DataFileHeader;

    MatrixDataset();
    MatrixDataset(const char * filename);
    MatrixDataset(int numelements, int width, int height, bool osh);
    ~MatrixDataset();
    int               CreateEmpty(int num, int wid, int heg, bool osh);
    int               Addelements(Matrix matrix, int numElements);
    int               SaveToFile(const char * filename);
    int               SaveToFile();
    void              showmatrix(bool result);

    bool              oshm;
    int               mNumelements;
    Matrix            mpMatrix;
private:

    fstream           mDatasetFile;
    char            * mpFilename;
    size_t            mDatasetFileSize;
};

#endif

