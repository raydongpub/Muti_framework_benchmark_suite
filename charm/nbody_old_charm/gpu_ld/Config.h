#ifndef _CONFIG_H_
#define _CONFIG_H_

#include <fstream>
#include <string>
#include <map>

#define MAX_TOK_LEN       0x0400
#define ECFG_MSG_LEN      0x00FF
#define ECFG_SUCCESS      0x0000
#define ECFG_OFILE        0x0001
#define ECFG_MEMALLOC     0x0002
#define ECFG_INVALID_PAIR 0x0003
#define ECFG_LONG_TOKEN   0x0004

#ifndef _DOUBLE_PRECISION
#define PRECISION float
#else
#define PRECISION double
#endif


using namespace std;

class Config {

public:
    Config(const char * filename);
    ~Config();
    const char        *  GetValue(const char * key);
    const char        *  GetValue(char * key);
    const char        *  GetValue(string key);
    static const char *  GetClassID();
    static const char *  GetEMSG(int errId);

private:
    typedef       map<string, string>   ConfigTable;
    typedef       ConfigTable::iterator ConfigIterator;
    ConfigTable   mConfigTable;

protected:
    int           ParseConfig();

    fstream       mFile;
    size_t        mFileSize;
    char        * mpContent;
    static char   mspClassID[];
    static char   mspEMSG[][ECFG_MSG_LEN];
};

#endif /* _CONFIG_H_ */

