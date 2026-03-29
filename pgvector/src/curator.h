#ifndef CURATOR_H
#define CURATOR_H

#include "postgres.h"

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "access/parallel.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "nodes/pg_list.h"
#include "utils/hsearch.h"
#include "utils/sampling.h"
#include "utils/tuplesort.h"
#include "vector.h"

/*
 * Curator is implemented as a hierarchical multi-tenant IVF index.
 *
 * The core algorithmic invariants intentionally mirror the original Curator
 * implementation and paper:
 * - a shared clustering tree trained recursively from the root
 * - a Bloom filter per node for tenant-aware pruning
 * - per-tenant shortlists that remain at higher nodes until they outgrow the
 *   shortlist threshold, at which point they are pushed down the tree
 * - exact vector verification over candidate buckets selected by best-first
 *   traversal of the clustering tree
 */

#define CURATOR_MAX_DIM 2000

/* Support functions: keep the same ordering contract as IVFFlat */
#define CURATOR_DISTANCE_PROC 1
#define CURATOR_NORM_PROC 2
#define CURATOR_KMEANS_DISTANCE_PROC 3
#define CURATOR_KMEANS_NORM_PROC 4
#define CURATOR_TYPE_INFO_PROC 5

#define CURATOR_VERSION        1
#define CURATOR_MAGIC_NUMBER   0xC812A701
#define CURATOR_PAGE_ID        0xFF95

/* Preserved page numbers */
#define CURATOR_METAPAGE_BLKNO 0
#define CURATOR_ROOT_BLKNO     1

/* Curator reloptions */
#define CURATOR_DEFAULT_LISTS         32
#define CURATOR_MIN_LISTS             1
#define CURATOR_MAX_LISTS             32768
#define CURATOR_DEFAULT_BF_CAPACITY   1000
#define CURATOR_MIN_BF_CAPACITY       1
#define CURATOR_DEFAULT_BF_FALSE_POS  0.01
#define CURATOR_MIN_BF_FALSE_POS      0.000001
#define CURATOR_MAX_BF_FALSE_POS      0.5
#define CURATOR_DEFAULT_MAX_SL_SIZE   128
#define CURATOR_MIN_MAX_SL_SIZE       1
#define CURATOR_DEFAULT_GAMMA1        16.0
#define CURATOR_DEFAULT_GAMMA2        256.0

/* Constants copied from the original Curator implementation */
#define CURATOR_MIN_POINTS_PER_CENTROID 8
#define CURATOR_MAX_LEVEL               8
#define CURATOR_N_CLUSTER_DIVISOR       2.0
#define CURATOR_MIN_N_CLUSTERS          4
#define CURATOR_BF_UPDATE_INTERVAL      100

/* Build phases */
#define PROGRESS_CURATOR_PHASE_KMEANS   2
#define PROGRESS_CURATOR_PHASE_ASSIGN   3
#define PROGRESS_CURATOR_PHASE_LOAD     4

/* Tuple / node kinds */
#define CURATOR_NODE_TUPLE_TYPE         1
#define CURATOR_VECTOR_TUPLE_TYPE       2

#define CURATOR_NODE_FLAG_ROOT          0x01
#define CURATOR_NODE_FLAG_LEAF          0x02

#define CuratorPageGetOpaque(page)      ((CuratorPageOpaque) PageGetSpecialPointer(page))
#define CuratorPageGetMeta(page)        ((CuratorMetaPageData *) PageGetContents(page))

/*
 * Labels in the original Curator code are external vector labels.
 * Inside PostgreSQL the external identity is the heap TID, while the search
 * pipeline benefits from a compact sequential internal vector id.
 */
typedef uint32 CuratorVid;
typedef int32 CuratorTenantId;

typedef struct CuratorTreeNodeData CuratorTreeNodeData;
typedef struct CuratorTreeNodeData *CuratorTreeNode;

typedef struct CuratorChildRefData
{
    BlockNumber blkno;
    OffsetNumber offno;
    uint16 reserved;
} CuratorChildRefData;

typedef struct CuratorLeafTenantCountData
{
    CuratorTenantId tenantId;
    uint32 count;
} CuratorLeafTenantCountData;

typedef struct CuratorShortlistHeaderData
{
    CuratorTenantId tenantId;
    uint32 length;
} CuratorShortlistHeaderData;

typedef struct CuratorBloomFilterData
{
    uint32 nbits;
    uint16 nhashes;
    uint16 reserved;
    uint8 data[FLEXIBLE_ARRAY_MEMBER];
} CuratorBloomFilterData;

typedef CuratorBloomFilterData *CuratorBloomFilter;

/*
 * Build-time vector store faithful to the original Curator implementation.
 * Vectors are materialized in contiguous float storage and referenced by vid.
 */
typedef struct CuratorVectorStoreData
{
    int dimensions;
    uint32 count;
    uint32 capacity;
    float *vecs;
} CuratorVectorStoreData;

typedef CuratorVectorStoreData *CuratorVectorStore;

/* heap TID -> internal vector id */
typedef struct CuratorLabelMapEntry
{
    ItemPointerData heapTid;
    CuratorVid vid;
} CuratorLabelMapEntry;

/* per-vid access list */
typedef struct CuratorAccessListData
{
    uint32 length;
    CuratorTenantId tenants[FLEXIBLE_ARRAY_MEMBER];
} CuratorAccessListData;

typedef CuratorAccessListData *CuratorAccessList;

typedef struct CuratorAccessEntry
{
    CuratorVid vid;
    CuratorAccessList access;
} CuratorAccessEntry;

/* per-node shortlist entry */
typedef struct CuratorShortlistEntry
{
    CuratorTenantId tenantId;
    uint32 nvids;
    CuratorVid *vids;
} CuratorShortlistEntry;

/* per-leaf tenant occupancy */
typedef struct CuratorTenantCountEntry
{
    CuratorTenantId tenantId;
    uint32 count;
} CuratorTenantCountEntry;

/* transient in-memory tree node used during build / search */
struct CuratorTreeNodeData
{
    uint8 level;
    uint8 flags;
    uint16 siblingId;
    uint16 nClusters;
    uint16 nChildren;

    CuratorTreeNode parent;
    CuratorTreeNode *children;

    Datum centroid;

    CuratorBloomFilter bloom;

    /* tenantId -> CuratorShortlistEntry */
    HTAB *shortlists;

    /* leaf-only state */
    CuratorVid *vectorIds;
    uint32 nVectorIds;
    uint32 maxVectorIds;

    /* tenantId -> CuratorTenantCountEntry */
    HTAB *tenantCounts;

    /* assigned after serialization to index pages */
    BlockNumber blkno;
    OffsetNumber offno;
};

/*
 * Variable-length node tuple persisted on disk.
 * Payload layout after the fixed header is:
 *   centroid bytes
 *   CuratorChildRefData[nChildren]
 *   CuratorBloomFilterData bytes
 *   CuratorShortlistHeaderData[nShortlists]
 *   CuratorVid[sum(shortlist lengths)]
 *   CuratorLeafTenantCountData[nTenantCounts]
 *   CuratorVid[nVectorIds]                  -- leaf nodes only
 */
typedef struct CuratorNodeTupleData
{
    uint8 type;
    uint8 level;
    uint8 flags;
    uint8 reserved;

    uint16 siblingId;
    uint16 nClusters;
    uint16 nChildren;
    uint16 nShortlists;
    uint16 nTenantCounts;
    uint16 nVectorIds;

    BlockNumber parentBlkno;
    OffsetNumber parentOffno;

    uint32 centroidSize;
    uint32 bloomSize;

    char payload[FLEXIBLE_ARRAY_MEMBER];
} CuratorNodeTupleData;

typedef CuratorNodeTupleData *CuratorNodeTuple;

/* leaf vector payload */
typedef struct CuratorVectorTupleData
{
    uint8 type;
    uint8 reserved1;
    uint16 accessCount;
    CuratorVid vid;
    ItemPointerData heapTid;
    uint32 vectorSize;
    char payload[FLEXIBLE_ARRAY_MEMBER];
} CuratorVectorTupleData;

typedef CuratorVectorTupleData *CuratorVectorTuple;

/* reloptions */
typedef struct CuratorOptions
{
    int32 vl_len_;
    int lists;
    int bfCapacity;
    double bfFalsePositiveRate;
    int maxShortlistSize;
    int tenantColumn;
} CuratorOptions;

typedef struct CuratorMetaPageData
{
    uint32 magicNumber;
    uint32 version;
    uint16 dimensions;
    uint16 lists;
    uint32 bfCapacity;
    double bfFalsePositiveRate;
    uint16 maxShortlistSize;
    uint16 rootLevel;
    AttrNumber vectorAttno;
    AttrNumber tenantAttno;
    BlockNumber rootBlkno;
    OffsetNumber rootOffno;
} CuratorMetaPageData;

typedef CuratorMetaPageData *CuratorMetaPage;

typedef struct CuratorPageOpaqueData
{
    BlockNumber nextblkno;
    uint16 unused;
    uint16 page_id;
} CuratorPageOpaqueData;

typedef CuratorPageOpaqueData *CuratorPageOpaque;

typedef struct CuratorTypeInfo
{
    int maxDimensions;
    Datum (*normalize) (PG_FUNCTION_ARGS);
    Size (*itemSize) (int dimensions);
    void (*checkValue) (Pointer value);
    void (*updateCenter) (Pointer value, int dimensions, float *accum);
    void (*sumCenter) (Pointer value, float *accum);
} CuratorTypeInfo;

typedef struct CuratorSupport
{
    FmgrInfo *procinfo;
    FmgrInfo *normprocinfo;
    FmgrInfo *kmeansprocinfo;
    FmgrInfo *kmeansnormprocinfo;
    Oid collation;
} CuratorSupport;

typedef struct CuratorBuildState
{
    Relation heap;
    Relation index;
    IndexInfo *indexInfo;
    TupleDesc tupdesc;
    const CuratorTypeInfo *typeInfo;

    int dimensions;
    int lists;
    int bfCapacity;
    double bfFalsePositiveRate;
    int maxShortlistSize;

    AttrNumber vectorAttno;
    AttrNumber tenantAttno;

    double indtuples;
    double reltuples;

    CuratorSupport support;

    CuratorTreeNode root;
    CuratorVectorStore vecStore;

    /* heap TID -> vid */
    HTAB *labelToVid;
    /* vid -> leaf node */
    HTAB *vidToLeaf;
    /* vid -> CuratorAccessEntry */
    HTAB *accessMatrix;

    uint32 nextVid;
    uint32 updateBloomAfter;

    MemoryContext buildCtx;
    MemoryContext tmpCtx;
} CuratorBuildState;

typedef struct CuratorCandidateBucket
{
    pairingheap_node phNode;
    BlockNumber blkno;
    OffsetNumber offno;
    double distance;
    uint32 nvecs;
} CuratorCandidateBucket;

typedef struct CuratorScanOpaqueData
{
    const CuratorTypeInfo *typeInfo;
    CuratorSupport support;

    bool first;
    Datum queryValue;
    CuratorTenantId tenantId;
    double gamma1;
    double gamma2;
    int dimensions;
    int requestedK;
    int resultCount;
    int resultIndex;
    float *distances;
    ItemPointerData *heapTids;
    MemoryContext tmpCtx;
} CuratorScanOpaqueData;

typedef CuratorScanOpaqueData *CuratorScanOpaque;

/* session-level search knobs */
extern int curator_tenant_id;
extern double curator_gamma1;
extern double curator_gamma2;

/* reloption accessors */
int CuratorGetLists(Relation index);
int CuratorGetBloomFilterCapacity(Relation index);
double CuratorGetBloomFilterFalsePositiveRate(Relation index);
int CuratorGetMaxShortlistSize(Relation index);
const char *CuratorGetTenantColumn(Relation index);
AttrNumber CuratorGetTenantAttributeNumber(Relation index, Relation heap, const char *columnName);

/* support / initialization */
FmgrInfo *CuratorOptionalProcInfo(Relation index, uint16 procnum);
Datum CuratorNormValue(const CuratorTypeInfo *typeInfo, Oid collation, Datum value);
bool CuratorCheckNorm(CuratorSupport *support, Datum value);
const CuratorTypeInfo *CuratorGetTypeInfo(Relation index);
void CuratorInit(void);
void CuratorInitPage(Buffer buf, Page page);
Buffer CuratorNewBuffer(Relation index, ForkNumber forkNum);
void CuratorInitRegisterPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state);
void CuratorCommitBuffer(Buffer buf, GenericXLogState *state);
void CuratorAppendPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state, ForkNumber forkNum);
void CuratorGetMetaPageInfo(Relation index, CuratorMetaPageData *meta);

/* Bloom filter helpers */
Size CuratorBloomFilterSize(uint32 nbits);
CuratorBloomFilter CuratorBloomFilterCreate(MemoryContext ctx, uint32 projectedCount, double falsePositiveRate);
void CuratorBloomFilterReset(CuratorBloomFilter filter);
void CuratorBloomFilterAdd(CuratorBloomFilter filter, CuratorTenantId tenantId);
bool CuratorBloomFilterMayContain(const CuratorBloomFilter filter, CuratorTenantId tenantId);
void CuratorBloomFilterUnionInto(CuratorBloomFilter dst, const CuratorBloomFilter src);

/* build-time faithful Curator helpers */
CuratorTreeNode CuratorCreateTreeNode(MemoryContext ctx, uint8 level, uint16 siblingId,
                                      CuratorTreeNode parent, Datum centroid,
                                      int dimensions, uint16 nClusters,
                                      uint32 bfCapacity, double bfFalsePositiveRate);
void CuratorTrainTree(CuratorBuildState *buildState, CuratorTreeNode node,
                      Datum *values, CuratorVid *vids, uint32 nvecs);
CuratorTreeNode CuratorAssignVectorToLeaf(CuratorTreeNode root, Datum value);
void CuratorGrantAccessAlongPath(CuratorBuildState *buildState, CuratorTreeNode node,
                                 CuratorVid vid, CuratorTenantId tenantId,
                                 const uint16 *path, uint8 pathLength);
void CuratorSplitShortlist(CuratorBuildState *buildState, CuratorTreeNode node,
                           CuratorTenantId tenantId);
bool CuratorMergeShortlist(CuratorTreeNode node, CuratorTenantId tenantId);
bool CuratorMergeShortlistRecursively(CuratorTreeNode node, CuratorTenantId tenantId);
void CuratorUpdateShortlistsAfterDelete(CuratorBuildState *buildState, CuratorTreeNode leaf,
                                        CuratorVid vid, const CuratorTenantId *tenantIds,
                                        uint32 ntenants);
void CuratorRefreshBloomFilters(CuratorBuildState *buildState, CuratorTreeNode leaf);

/* scan helpers */
void CuratorSearch(Relation index, Datum queryValue, int k, CuratorTenantId tenantId,
                   float *distances, ItemPointerData *heapTids, MemoryContext tmpCtx);
void CuratorSearchOne(Relation index, Datum queryValue, int k, CuratorTenantId tenantId,
                      float *distances, ItemPointerData *heapTids, MemoryContext tmpCtx);

/* index access method callbacks */
IndexBuildResult *curatorbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void curatorbuildempty(Relation index);
bool curatorinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
                   Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
                   , bool indexUnchanged
#endif
                   , IndexInfo *indexInfo);
IndexBulkDeleteResult *curatorbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
                                         IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *curatorvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc curatorbeginscan(Relation index, int nkeys, int norderbys);
void curatorrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool curatorgettuple(IndexScanDesc scan, ScanDirection dir);
void curatorendscan(IndexScanDesc scan);

#endif
