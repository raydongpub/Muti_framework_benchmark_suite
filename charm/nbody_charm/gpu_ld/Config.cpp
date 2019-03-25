#include "Config.h"
#include <ctype.h>
#include <string.h>
#include <string>
#include <iostream>

Config::Config(const char * filename) {

    int errId;

    mFile.open(filename, ios_base::in);
    if (!(mFile.is_open()))
        throw (int) ECFG_OFILE;

    mFile.seekg(0, ios_base::end);
    mFileSize = mFile.tellg();
    mFile.seekg(0, ios_base::beg);

    try {
        mpContent = new char[mFileSize];
    }
    catch (bad_alloc & e) {
        throw (int) ECFG_MEMALLOC;
    }

    mFile.read(mpContent, mFileSize);
    mFile.close();
    if ((errId = ParseConfig()) != ECFG_SUCCESS)
        throw (int) errId;
}

Config::~Config() {

    delete [] mpContent;
}

const char * Config::GetValue(const char * key) {
    char   * c_key = const_cast<char *>(key);
    string   s_key = string(c_key);

    return GetValue(s_key);
}

const char * Config::GetValue(char * key) {
    string s_key = string(key);

    return GetValue(s_key);
}

const char * Config::GetValue(string key) {

    string         key_cap;
    ConfigIterator it;
    int            idx, len;

    key_cap = key;
    len = key_cap.length();
    for (idx=0;idx<len;idx++)
        key_cap[idx] = toupper(key_cap[idx]);

    it = mConfigTable.find(key_cap);
    if (it == mConfigTable.end())
        return NULL;

    return (it->second).c_str();
}

int Config::ParseConfig() {
    typedef enum {
        PARSE_INACTIVE = 0,
        PARSE_ACTIVE   = 1,
        PARSE_KEYTOK   = 2,
        PARSE_VALTOK   = 3
    } ParseState;

#define TOK_CRLF    '\n'
#define TOK_SPC     ' '
#define TOK_POUND   '#'
#define TOK_TAB     '\t'
#define _CheckBufferLimit(str)   \
    if (str##Idx >= MAX_TOK_LEN) \
        return ECFG_LONG_TOKEN

    ParseState state;
    size_t     idx;
    char       key[MAX_TOK_LEN], val[MAX_TOK_LEN];
    int        keyIdx, valIdx, len, jIdx;
    char       c;

    memset(key, 0, MAX_TOK_LEN);
    memset(val, 0, MAX_TOK_LEN);

    state  = PARSE_ACTIVE;
    keyIdx = 0;
    valIdx = 0;

    for (idx=0;idx<mFileSize;idx++) {
        c = mpContent[idx];
        switch (state) {
        case PARSE_INACTIVE:
            switch (c) {
            case TOK_CRLF:
                state = PARSE_ACTIVE;
                break;
            case TOK_SPC:
                break;
            case TOK_POUND:
                break;
            case TOK_TAB:
                break;
            default:
                break;
            }
            break;

        case PARSE_ACTIVE:
            switch (c) {
            case TOK_CRLF:
                break;
            case TOK_SPC:
                break;
            case TOK_POUND:
                state         = PARSE_INACTIVE;
                break;
            case TOK_TAB:
                break;
            default:
                state         = PARSE_KEYTOK;
                keyIdx        = 0;
                valIdx        = 0;
                key[keyIdx++] = c;
                _CheckBufferLimit(key);
                break;
            }
            break;

        case PARSE_KEYTOK:
            switch (c) {
            case TOK_CRLF:
                state         = PARSE_VALTOK;
                key[keyIdx]   = 0x00;
                break;
            case TOK_SPC:
                state         = PARSE_VALTOK;
                key[keyIdx]   = 0x00;
                break;
            case TOK_POUND:
                return ECFG_INVALID_PAIR;
                break;
            case TOK_TAB:
                state         = PARSE_VALTOK;
                key[keyIdx]   = 0x00;
                break;
            default:
                key[keyIdx++] = c;
                _CheckBufferLimit(key);
                break;
            }
            break;

        case PARSE_VALTOK:
            switch (c) {
            case TOK_CRLF:
                state         = PARSE_ACTIVE;
                val[valIdx]   = 0x00;

//                    len           = valIdx;
//                    for (jIdx=0;jIdx<len;jIdx++)
//                        val[jIdx] = toupper(val[jIdx]);
                    len           = keyIdx;
                    for (jIdx=0;jIdx<len;jIdx++)
                        key[jIdx] = toupper(key[jIdx]);
                    
                mConfigTable.insert(make_pair(string(key), string(val)));
                break;
            case TOK_SPC:
                break;
            case TOK_POUND:
                if (!valIdx)
                    return ECFG_INVALID_PAIR;
                else {
                    state         = PARSE_INACTIVE;
                    val[valIdx]   = 0x00;

//                    len           = valIdx;
//                    for (jIdx=0;jIdx<len;jIdx++)
//                        val[jIdx] = toupper(val[jIdx]);
                    len           = keyIdx;
                    for (jIdx=0;jIdx<len;jIdx++)
                        key[jIdx] = toupper(key[jIdx]);
                    
                    mConfigTable.insert(make_pair(string(key), string(val)));
                }
                break;
            case TOK_TAB:
                break;
            default:
                val[valIdx++] = c;
                _CheckBufferLimit(val);
                break;
            }
            break;

        default:
            break;
        }
    }

    return ECFG_SUCCESS;

#undef TOK_CRLF
#undef TOK_SPC
#undef TOK_POUND
#undef TOK_TAB
}

const char * Config::GetClassID() {
    return Config::mspClassID;
}

const char * Config::GetEMSG(int errId) {
    return Config::mspEMSG[errId];
}

char Config::mspClassID[] = "Config";

char Config::mspEMSG[][ECFG_MSG_LEN] = {
    "Success",
    "Fail to read input file",
    "Fail to allocate memory",
    "Invalid key-value pair",
    "Key/Value too long"
};

