#include "postgres.h"

#include <float.h>

#include "access/amapi.h"
#include "access/reloptions.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "curator.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/float.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

int curator_tenant_id = -1;
double curator_gamma1;
double curator_gamma2;

static relopt_kind curator_relopt_kind;

PGDLLEXPORT Datum l2_normalize(PG_FUNCTION_ARGS);

static Size
VectorItemSize(int dimensions)
{
    return VECTOR_SIZE(dimensions);
}

static void
VectorUpdateCenter(Pointer value, int dimensions, float *accum)
{
    Vector *vec = (Vector *) value;

    SET_VARSIZE(vec, VECTOR_SIZE(dimensions));
    vec->dim = dimensions;

    for (int i = 0; i < dimensions; i++)
        vec->x[i] = accum[i];
}

static void
VectorSumCenter(Pointer value, float *accum)
{
    Vector *vec = (Vector *) value;

    for (int i = 0; i < vec->dim; i++)
        accum[i] += vec->x[i];
}

static void
curator_tenant_column_validator(const char *value)
{
    if (value == NULL || value[0] == '\0')
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("tenant_column cannot be empty")));

    if (strlen(value) >= NAMEDATALEN)
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("tenant_column must be shorter than %d bytes", NAMEDATALEN)));
}

void
CuratorInit(void)
{
    curator_relopt_kind = add_reloption_kind();

    add_int_reloption(curator_relopt_kind, "lists", "Number of root clusters",
                      CURATOR_DEFAULT_LISTS, CURATOR_MIN_LISTS, CURATOR_MAX_LISTS,
                      AccessExclusiveLock);
    add_int_reloption(curator_relopt_kind, "bf_capacity", "Projected element count for node Bloom filters",
                      CURATOR_DEFAULT_BF_CAPACITY, CURATOR_MIN_BF_CAPACITY, INT_MAX,
                      AccessExclusiveLock);
    add_real_reloption(curator_relopt_kind, "bf_false_pos", "False positive rate for node Bloom filters",
                       CURATOR_DEFAULT_BF_FALSE_POS, CURATOR_MIN_BF_FALSE_POS,
                       CURATOR_MAX_BF_FALSE_POS, AccessExclusiveLock);
    add_int_reloption(curator_relopt_kind, "max_sl_size", "Maximum shortlist size before split",
                      CURATOR_DEFAULT_MAX_SL_SIZE, CURATOR_MIN_MAX_SL_SIZE, INT_MAX,
                      AccessExclusiveLock);
    add_string_reloption(curator_relopt_kind, "tenant_column",
                         "Name of the int[] column that stores tenant access lists",
                         "access_tenants", curator_tenant_column_validator,
                         AccessExclusiveLock);

    DefineCustomIntVariable("curator.tenant_id", "Sets the tenant id used by curator scans",
                            "Must be set by the query path before an index scan.",
                            &curator_tenant_id,
                            -1, -1, INT_MAX, PGC_USERSET, 0,
                            NULL, NULL, NULL);

    DefineCustomRealVariable("curator.gamma1", "Sets Curator's first candidate expansion factor",
                             NULL, &curator_gamma1,
                             CURATOR_DEFAULT_GAMMA1, 1.0, DBL_MAX, PGC_USERSET, 0,
                             NULL, NULL, NULL);

    DefineCustomRealVariable("curator.gamma2", "Sets Curator's second candidate truncation factor",
                             NULL, &curator_gamma2,
                             CURATOR_DEFAULT_GAMMA2, 1.0, DBL_MAX, PGC_USERSET, 0,
                             NULL, NULL, NULL);

    MarkGUCPrefixReserved("curator");
}

static char *
curatorbuildphasename(int64 phasenum)
{
    switch (phasenum)
    {
        case PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE:
            return "initializing";
        case PROGRESS_CURATOR_PHASE_KMEANS:
            return "performing hierarchical k-means";
        case PROGRESS_CURATOR_PHASE_ASSIGN:
            return "assigning tuples to tree leaves";
        case PROGRESS_CURATOR_PHASE_LOAD:
            return "loading tuples";
        default:
            return NULL;
    }
}

static void
curatorcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
                    Cost *indexStartupCost, Cost *indexTotalCost,
                    Selectivity *indexSelectivity, double *indexCorrelation,
                    double *indexPages)
{
    GenericCosts costs;
    int lists;
    double ratio;
    double sequentialRatio = 0.5;
    double startupPages;
    double spc_seq_page_cost;
    Relation index;
    double effectiveGamma2;

    if (path->indexorderbys == NIL)
    {
        *indexStartupCost = get_float8_infinity();
        *indexTotalCost = get_float8_infinity();
        *indexSelectivity = 0;
        *indexCorrelation = 0;
        *indexPages = 0;
#if PG_VERSION_NUM >= 180000
        path->path.disabled_nodes = 2;
#endif
        return;
    }

    MemSet(&costs, 0, sizeof(costs));
    genericcostestimate(root, path, loop_count, &costs);

    index = index_open(path->indexinfo->indexoid, NoLock);
    lists = CuratorGetLists(index);
    index_close(index, NoLock);

    effectiveGamma2 = curator_gamma2 > 1.0 ? curator_gamma2 : 1.0;
    ratio = effectiveGamma2 / (double) Max(1, lists);
    if (ratio > 1.0)
        ratio = 1.0;

    get_tablespace_page_costs(path->indexinfo->reltablespace, NULL, &spc_seq_page_cost);

    costs.indexTotalCost -= sequentialRatio * costs.numIndexPages *
        (costs.spc_random_page_cost - spc_seq_page_cost);
    costs.indexStartupCost = costs.indexTotalCost * ratio;

    startupPages = costs.numIndexPages * ratio;
    if (startupPages > path->indexinfo->rel->pages && ratio < 0.5)
    {
        costs.indexStartupCost -= (1 - sequentialRatio) * startupPages *
            (costs.spc_random_page_cost - spc_seq_page_cost);
        costs.indexStartupCost -= (startupPages - path->indexinfo->rel->pages) * spc_seq_page_cost;
    }

    *indexStartupCost = costs.indexStartupCost;
    *indexTotalCost = costs.indexTotalCost;
    *indexSelectivity = costs.indexSelectivity;
    *indexCorrelation = costs.indexCorrelation;
    *indexPages = costs.numIndexPages;
}

static bytea *
curatoroptions(Datum reloptions, bool validate)
{
    static const relopt_parse_elt tab[] = {
        {"lists", RELOPT_TYPE_INT, offsetof(CuratorOptions, lists)},
        {"bf_capacity", RELOPT_TYPE_INT, offsetof(CuratorOptions, bfCapacity)},
        {"bf_false_pos", RELOPT_TYPE_REAL, offsetof(CuratorOptions, bfFalsePositiveRate)},
        {"max_sl_size", RELOPT_TYPE_INT, offsetof(CuratorOptions, maxShortlistSize)},
        {"tenant_column", RELOPT_TYPE_STRING, offsetof(CuratorOptions, tenantColumn)},
    };

    return (bytea *) build_reloptions(reloptions, validate,
                                      curator_relopt_kind,
                                      sizeof(CuratorOptions),
                                      tab, lengthof(tab));
}

static bool
curatorvalidate(Oid opclassoid)
{
    return true;
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(curatorhandler);
Datum
curatorhandler(PG_FUNCTION_ARGS)
{
    IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

    amroutine->amstrategies = 0;
    amroutine->amsupport = 5;
    amroutine->amoptsprocnum = 0;
    amroutine->amcanorder = false;
    amroutine->amcanorderbyop = true;
#if PG_VERSION_NUM >= 180000
    amroutine->amcanhash = false;
    amroutine->amconsistentequality = false;
    amroutine->amconsistentordering = false;
#endif
    amroutine->amcanbackward = false;
    amroutine->amcanunique = false;
    amroutine->amcanmulticol = false;
    amroutine->amoptionalkey = true;
    amroutine->amsearcharray = false;
    amroutine->amsearchnulls = false;
    amroutine->amstorage = false;
    amroutine->amclusterable = false;
    amroutine->ampredlocks = false;
    amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
    amroutine->amcanbuildparallel = false;
#endif
    amroutine->amcaninclude = false;
    amroutine->amusemaintenanceworkmem = false;
#if PG_VERSION_NUM >= 160000
    amroutine->amsummarizing = false;
#endif
    amroutine->amparallelvacuumoptions = VACUUM_OPTION_NO_PARALLEL;
    amroutine->amkeytype = InvalidOid;

    amroutine->ambuild = curatorbuild;
    amroutine->ambuildempty = curatorbuildempty;
    amroutine->aminsert = curatorinsert;
#if PG_VERSION_NUM >= 170000
    amroutine->aminsertcleanup = NULL;
#endif
    amroutine->ambulkdelete = curatorbulkdelete;
    amroutine->amvacuumcleanup = curatorvacuumcleanup;
    amroutine->amcanreturn = NULL;
    amroutine->amcostestimate = curatorcostestimate;
#if PG_VERSION_NUM >= 180000
    amroutine->amgettreeheight = NULL;
#endif
    amroutine->amoptions = curatoroptions;
    amroutine->amproperty = NULL;
    amroutine->ambuildphasename = curatorbuildphasename;
    amroutine->amvalidate = curatorvalidate;
#if PG_VERSION_NUM >= 140000
    amroutine->amadjustmembers = NULL;
#endif
    amroutine->ambeginscan = curatorbeginscan;
    amroutine->amrescan = curatorrescan;
    amroutine->amgettuple = curatorgettuple;
    amroutine->amgetbitmap = NULL;
    amroutine->amendscan = curatorendscan;
    amroutine->ammarkpos = NULL;
    amroutine->amrestrpos = NULL;

    amroutine->amestimateparallelscan = NULL;
    amroutine->aminitparallelscan = NULL;
    amroutine->amparallelrescan = NULL;

#if PG_VERSION_NUM >= 180000
    amroutine->amtranslatestrategy = NULL;
    amroutine->amtranslatecmptype = NULL;
#endif

    PG_RETURN_POINTER(amroutine);
}

int
CuratorGetLists(Relation index)
{
    CuratorOptions *opts = (CuratorOptions *) index->rd_options;

    if (opts != NULL)
        return opts->lists;

    return CURATOR_DEFAULT_LISTS;
}

int
CuratorGetBloomFilterCapacity(Relation index)
{
    CuratorOptions *opts = (CuratorOptions *) index->rd_options;

    if (opts != NULL)
        return opts->bfCapacity;

    return CURATOR_DEFAULT_BF_CAPACITY;
}

double
CuratorGetBloomFilterFalsePositiveRate(Relation index)
{
    CuratorOptions *opts = (CuratorOptions *) index->rd_options;

    if (opts != NULL)
        return opts->bfFalsePositiveRate;

    return CURATOR_DEFAULT_BF_FALSE_POS;
}

int
CuratorGetMaxShortlistSize(Relation index)
{
    CuratorOptions *opts = (CuratorOptions *) index->rd_options;

    if (opts != NULL)
        return opts->maxShortlistSize;

    return CURATOR_DEFAULT_MAX_SL_SIZE;
}

const char *
CuratorGetTenantColumn(Relation index)
{
    CuratorOptions *opts = (CuratorOptions *) index->rd_options;
    const char *tenantColumn = NULL;

    if (opts != NULL)
        tenantColumn = GET_STRING_RELOPTION(opts, tenantColumn);

    return tenantColumn != NULL ? tenantColumn : "access_tenants";
}

AttrNumber
CuratorGetTenantAttributeNumber(Relation index, Relation heap, const char *columnName)
{
    TupleDesc tupdesc = RelationGetDescr(heap);
    const char *target = columnName != NULL ? columnName : CuratorGetTenantColumn(index);

    for (int attno = 0; attno < tupdesc->natts; attno++)
    {
        Form_pg_attribute attr = TupleDescAttr(tupdesc, attno);

        if (attr->attisdropped)
            continue;

        if (strcmp(NameStr(attr->attname), target) == 0)
            return attr->attnum;
    }

    ereport(ERROR,
            (errcode(ERRCODE_UNDEFINED_COLUMN),
             errmsg("tenant column \"%s\" does not exist", target)));
    return InvalidAttrNumber;
}

FmgrInfo *
CuratorOptionalProcInfo(Relation index, uint16 procnum)
{
    if (!OidIsValid(index_getprocid(index, 1, procnum)))
        return NULL;

    return index_getprocinfo(index, 1, procnum);
}

Datum
CuratorNormValue(const CuratorTypeInfo *typeInfo, Oid collation, Datum value)
{
    if (typeInfo == NULL || typeInfo->normalize == NULL)
        return value;

    return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

bool
CuratorCheckNorm(CuratorSupport *support, Datum value)
{
    if (support->normprocinfo == NULL)
        return true;

    return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo,
                                            support->collation,
                                            value)) > 0;
}

const CuratorTypeInfo *
CuratorGetTypeInfo(Relation index)
{
    FmgrInfo *procinfo = CuratorOptionalProcInfo(index, CURATOR_TYPE_INFO_PROC);

    if (procinfo == NULL)
    {
        static const CuratorTypeInfo typeInfo = {
            .maxDimensions = CURATOR_MAX_DIM,
            .normalize = l2_normalize,
            .itemSize = VectorItemSize,
            .checkValue = NULL,
            .updateCenter = VectorUpdateCenter,
            .sumCenter = VectorSumCenter
        };

        return &typeInfo;
    }

    return (const CuratorTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

Buffer
CuratorNewBuffer(Relation index, ForkNumber forkNum)
{
    Buffer buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

    LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
    return buf;
}

void
CuratorInitPage(Buffer buf, Page page)
{
    PageInit(page, BufferGetPageSize(buf), sizeof(CuratorPageOpaqueData));
    CuratorPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
    CuratorPageGetOpaque(page)->page_id = CURATOR_PAGE_ID;
}

void
CuratorInitRegisterPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state)
{
    *state = GenericXLogStart(index);
    *page = GenericXLogRegisterBuffer(*state, *buf, GENERIC_XLOG_FULL_IMAGE);
    CuratorInitPage(*buf, *page);
}

void
CuratorCommitBuffer(Buffer buf, GenericXLogState *state)
{
    GenericXLogFinish(state);
    UnlockReleaseBuffer(buf);
}

void
CuratorAppendPage(Relation index, Buffer *buf, Page *page, GenericXLogState **state, ForkNumber forkNum)
{
    Buffer newbuf = CuratorNewBuffer(index, forkNum);
    Page newpage = GenericXLogRegisterBuffer(*state, newbuf, GENERIC_XLOG_FULL_IMAGE);

    CuratorPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);
    CuratorInitPage(newbuf, newpage);

    GenericXLogFinish(*state);
    UnlockReleaseBuffer(*buf);

    *state = GenericXLogStart(index);
    *page = GenericXLogRegisterBuffer(*state, newbuf, GENERIC_XLOG_FULL_IMAGE);
    *buf = newbuf;
}

void
CuratorGetMetaPageInfo(Relation index, CuratorMetaPageData *meta)
{
    Buffer buf;
    Page page;
    CuratorMetaPage metap;

    buf = ReadBuffer(index, CURATOR_METAPAGE_BLKNO);
    LockBuffer(buf, BUFFER_LOCK_SHARE);
    page = BufferGetPage(buf);
    metap = CuratorPageGetMeta(page);

    if (unlikely(metap->magicNumber != CURATOR_MAGIC_NUMBER))
        elog(ERROR, "curator index is not valid");

    if (meta != NULL)
        memcpy(meta, metap, sizeof(CuratorMetaPageData));

    UnlockReleaseBuffer(buf);
}
