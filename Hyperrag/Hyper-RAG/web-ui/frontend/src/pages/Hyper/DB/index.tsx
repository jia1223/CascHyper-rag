import React, { useState, useEffect } from 'react';
import {
    Table,
    Button,
    Modal,
    Form,
    Input,
    Select,
    message,
    Space,
    Tabs,
    Popconfirm,
    Tag,
    Descriptions,
    Card,
    Spin,
    Tooltip,
    Typography,
    Alert,
    Row,
    Col,
    Statistic
} from 'antd';
import {
    PlusOutlined,
    EditOutlined,
    DeleteOutlined,
    EyeOutlined,
    DatabaseOutlined
} from '@ant-design/icons';
import { observer } from 'mobx-react';
import { useTranslation } from 'react-i18next';
import { storeGlobalUser } from '../../../store/globalUser';
import DatabaseSelector from '../../../components/DatabaseSelector';
import { SERVER_URL } from '../../../utils';

const { Text } = Typography;
import HyperGraph from '../../../components/HyperGraph';

const { TextArea } = Input;
const { Option } = Select;
const { TabPane } = Tabs;

const HyperDB = () => {
    const { t } = useTranslation();
    const [vertices, setVertices] = useState([]);
    const [hyperedges, setHyperedges] = useState([]);
    const [loading, setLoading] = useState(false);
    const [modalVisible, setModalVisible] = useState(false);
    const [modalLoading, setModalLoading] = useState(false);
    const [modalType, setModalType] = useState('add'); // add, edit, view
    const [modalDataType, setModalDataType] = useState('vertex'); // vertex, hyperedge
    const [selectedRecord, setSelectedRecord] = useState(null);
    const [form] = Form.useForm();
    const [isVertexModalVisible, setIsVertexModalVisible] = useState(false);
    const [isHyperedgeModalVisible, setIsHyperedgeModalVisible] = useState(false);
    const [editingVertex, setEditingVertex] = useState(null);
    const [editingHyperedge, setEditingHyperedge] = useState(null);
    const [vertexForm] = Form.useForm();
    const [hyperedgeForm] = Form.useForm();

    // 分页相关状态
    const [verticesPagination, setVerticesPagination] = useState({
        current: 1,
        pageSize: 10,
        total: 0
    });
    const [hyperedgesPagination, setHyperedgesPagination] = useState({
        current: 1,
        pageSize: 10,
        total: 0
    });

    // 初始化数据库
    useEffect(() => {
        storeGlobalUser.restoreSelectedDatabase();
        storeGlobalUser.loadDatabases();
    }, []);

    // 获取数据
    const fetchData = async (database = storeGlobalUser.selectedDatabase, resetPagination = false) => {
        if (!database) {
return;
}

        setLoading(true);
        console.log('Fetching data from:', SERVER_URL, 'Database:', database);

        // 如果需要重置分页，则重置到第一页
        const currentVerticesPage = resetPagination ? 1 : verticesPagination.current;
        const currentHyperedgesPage = resetPagination ? 1 : hyperedgesPagination.current;

        try {
            const verticesUrl = `${SERVER_URL}/db/vertices?database=${encodeURIComponent(database)}&page=${currentVerticesPage}&page_size=${verticesPagination.pageSize}`;
            const hyperedgesUrl = `${SERVER_URL}/db/hyperedges?database=${encodeURIComponent(database)}&page=${currentHyperedgesPage}&page_size=${hyperedgesPagination.pageSize}`;

            console.log('Making requests to:');
            console.log('- Vertices:', verticesUrl);
            console.log('- Hyperedges:', hyperedgesUrl);

            const [verticesRes, hyperedgesRes] = await Promise.all([
                fetch(verticesUrl),
                fetch(hyperedgesUrl)
            ]);

            console.log('Response status:');
            console.log('- Vertices:', verticesRes.status, verticesRes.ok);
            console.log('- Hyperedges:', hyperedgesRes.status, hyperedgesRes.ok);

            if (!verticesRes.ok) {
                throw new Error(`Vertices API failed: ${verticesRes.status} ${verticesRes.statusText}`);
            }
            if (!hyperedgesRes.ok) {
                throw new Error(`Hyperedges API failed: ${hyperedgesRes.status} ${hyperedgesRes.statusText}`);
            }

            const verticesData = await verticesRes.json();
            const hyperedgesData = await hyperedgesRes.json();

            console.log('Data received:');
            console.log('- Vertices:', verticesData);
            console.log('- Hyperedges:', hyperedgesData);

            // 处理vertices数据
            if (verticesData.data) {
                // 分页数据
                setVertices(verticesData.data);
                setVerticesPagination({
                    current: verticesData.page,
                    pageSize: verticesData.page_size,
                    total: verticesData.total
                });
            } else {
                // 旧版本非分页数据
                setVertices(verticesData);
                setVerticesPagination({
                    current: 1,
                    pageSize: verticesData.length,
                    total: verticesData.length
                });
            }

            // 处理hyperedges数据
            if (hyperedgesData.data) {
                // 分页数据
                setHyperedges(hyperedgesData.data);
                setHyperedgesPagination({
                    current: hyperedgesData.page,
                    pageSize: hyperedgesData.page_size,
                    total: hyperedgesData.total
                });
            } else {
                // 旧版本非分页数据
                setHyperedges(hyperedgesData);
                setHyperedgesPagination({
                    current: 1,
                    pageSize: hyperedgesData.length,
                    total: hyperedgesData.length
                });
            }
        } catch (error) {
            console.error('Fetch error:', error);
            message.error(t('database.fetch_data_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 当数据库选择改变时，重新获取数据
    useEffect(() => {
        if (storeGlobalUser.selectedDatabase) {
            fetchData(storeGlobalUser.selectedDatabase, true);
        }
    }, [storeGlobalUser.selectedDatabase]);

    // 获取vertex详细信息
    const getVertexDetail = async (vertexId) => {
        try {
            const url = `${SERVER_URL}/db/vertices/${encodeURIComponent(vertexId)}?database=${encodeURIComponent(storeGlobalUser.selectedDatabase)}`;
            const response = await fetch(url);
            const data = await response.json();
            return data;
        } catch (error) {
            message.error(t('database.fetch_vertex_detail_failed') + '：' + error.message);
            return null;
        }
    };

    // 获取hyperedge详细信息
    const getHyperedgeDetail = async (hyperedgeId) => {
        try {
            const url = `${SERVER_URL}/db/hyperedges/${encodeURIComponent(hyperedgeId)}?database=${encodeURIComponent(storeGlobalUser.selectedDatabase)}`;
            const response = await fetch(url);
            const data = await response.json();
            return data;
        } catch (error) {
            message.error(t('database.fetch_hyperedge_detail_failed') + '：' + error.message);
            return null;
        }
    };

    // 添加/编辑vertex
    const handleVertexSubmit = async (values) => {
        try {
            setLoading(true);
            let response;

            const submitData = {
                ...values,
                database: storeGlobalUser.selectedDatabase
            };

            if (modalType === 'add') {
                response = await fetch(`${SERVER_URL}/db/vertices`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(submitData),
                });
            } else {
                const url = `${SERVER_URL}/db/vertices/${encodeURIComponent(selectedRecord)}?database=${encodeURIComponent(storeGlobalUser.selectedDatabase)}`;
                response = await fetch(url, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(submitData),
                });
            }

            const result = await response.json();

            if (result.success) {
                message.success(result.message);
                setModalVisible(false);
                form.resetFields();
                // 保持当前分页状态，不重置分页
                fetchData(storeGlobalUser.selectedDatabase, false);
            } else {
                message.error(result.message);
            }
        } catch (error) {
            message.error(t('database.operation_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 添加/编辑hyperedge
    const handleHyperedgeSubmit = async (values) => {
        try {
            setLoading(true);

            // 处理vertices字符串，转换为数组
            const verticesArray = values.vertices.split(',').map(v => v.trim()).filter(v => v);
            const submitData = {
                ...values,
                vertices: verticesArray,
                database: storeGlobalUser.selectedDatabase
            };

            let response;

            if (modalType === 'add') {
                response = await fetch(`${SERVER_URL}/db/hyperedges`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(submitData),
                });
            } else {
                response = await fetch(`${SERVER_URL}/db/hyperedges/${encodeURIComponent(selectedRecord)}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        keywords: values.keywords,
                        summary: values.summary
                    }),
                });
            }

            const result = await response.json();

            if (result.success) {
                message.success(result.message);
                setModalVisible(false);
                form.resetFields();
                // 保持当前分页状态，不重置分页
                fetchData(storeGlobalUser.selectedDatabase, false);
            } else {
                message.error(result.message);
            }
        } catch (error) {
            message.error(t('database.operation_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 删除vertex
    const handleDeleteVertex = async (vertexId) => {
        try {
            setLoading(true);
            const url = `${SERVER_URL}/db/vertices/${encodeURIComponent(vertexId)}?database=${encodeURIComponent(storeGlobalUser.selectedDatabase)}`;
            const response = await fetch(url, {
                method: 'DELETE',
            });

            const result = await response.json();

            if (result.success) {
                message.success(result.message);
                // 保持当前分页状态，不重置分页
                fetchData(storeGlobalUser.selectedDatabase, false);
            } else {
                message.error(result.message);
            }
        } catch (error) {
            message.error(t('database.delete_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 删除hyperedge
    const handleDeleteHyperedge = async (hyperedgeId) => {
        try {
            setLoading(true);
            const url = `${SERVER_URL}/db/hyperedges/${encodeURIComponent(hyperedgeId)}?database=${encodeURIComponent(storeGlobalUser.selectedDatabase)}`;
            const response = await fetch(url, {
                method: 'DELETE',
            });

            const result = await response.json();

            if (result.success) {
                message.success(result.message);
                // 保持当前分页状态，不重置分页
                fetchData(storeGlobalUser.selectedDatabase, false);
            } else {
                message.error(result.message);
            }
        } catch (error) {
            message.error(t('database.delete_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 打开modal
    const openModal = async (type, dataType, record = null) => {
        setModalType(type);
        setModalDataType(dataType);
        setSelectedRecord(record);

        // 重置表单
        form.resetFields();

        if ((type === 'edit' || type === 'view') && record) {
            if (dataType === 'vertex') {
                // 先显示modal，然后加载数据
                setModalVisible(true);
                setModalLoading(true);

                try {
                    const data = await getVertexDetail(record);
                    if (data) {
                        form.setFieldsValue({
                            vertex_id: record,
                            entity_name: data.entity_name || '',
                            entity_type: data.entity_type || '',
                            description: data.description || '',
                            additional_properties: data.additional_properties || ''
                        });
                    }
                } catch (error) {
                    message.error(t('database.fetch_vertex_detail_failed') + '：' + error.message);
                } finally {
                    setModalLoading(false);
                }
            } else if (dataType === 'hyperedge') {
                if (type === 'view') {
                    // 查看hyperedge详情
                    setModalVisible(true);
                    setModalLoading(true);

                    try {
                        const data = await getHyperedgeDetail(record);
                        if (data) {
                            form.setFieldsValue({
                                vertices: record.replace(/\|#\|/g, ', '),
                                keywords: data.keywords || '',
                                summary: data.summary || ''
                            });
                        }
                    } catch (error) {
                        message.error(t('database.fetch_hyperedge_detail_failed') + '：' + error.message);
                    } finally {
                        setModalLoading(false);
                    }
                } else {
                    // hyperedge编辑时，使用现有数据
                    setModalVisible(true);
                    setModalLoading(true);

                    try {
                        const data = await getHyperedgeDetail(record);
                        if (data) {
                            form.setFieldsValue({
                                vertices: record.replace(/\|#\|/g, ', '),
                                keywords: data.keywords || '',
                                summary: data.summary || ''
                            });
                        }
                    } catch (error) {
                        message.error(t('database.fetch_hyperedge_detail_failed') + '：' + error.message);
                    } finally {
                        setModalLoading(false);
                    }
                }
            }
        } else {
            // 添加模式直接显示modal
            setModalVisible(true);
        }
    };

    // Vertices表格列定义
    const vertexColumns = [
        {
            title: t('database.vertex_id'),
            dataIndex: 'vertex_id',
            key: 'vertex_id',
            width: 200,
            render: (text) => <Tag color="blue">{text}</Tag>,
        },
        {
            title: t('database.action'),
            key: 'action',
            width: 100,
            render: (_, record) => (
                <Space size="small">
                    <Button
                        type="link"
                        icon={<EyeOutlined />}
                        onClick={() => openModal('view', 'vertex', record.vertex_id)}
                    >
                        {t('database.view')}
                    </Button>
                    <Button
                        type="link"
                        icon={<EditOutlined />}
                        onClick={() => openModal('edit', 'vertex', record.vertex_id)}
                    >
                        {t('database.edit')}
                    </Button>
                    <Popconfirm
                        title={t('database.confirm_delete_vertex')}
                        onConfirm={() => handleDeleteVertex(record.vertex_id)}
                        okText={t('database.confirm')}
                        cancelText={t('database.cancel')}
                    >
                        <Button type="link" danger icon={<DeleteOutlined />}>
                            {t('database.delete')}
                        </Button>
                    </Popconfirm>
                </Space>
            ),
        },
    ];

    // Hyperedges表格列定义
    const hyperedgeColumns = [
        {
            title: t('database.hyperedge'),
            dataIndex: 'hyperedge_id',
            key: 'hyperedge_id',
            width: 200,
            render: (text, record) => {
                const vertices = text.split('|*|');
                const tooltipContent = (
                    <div>
                        <div><strong>{t('database.vertex_count')}:</strong> {vertices.length}</div>
                        {record.keywords && <div><strong>{t('database.keywords')}:</strong> {record.keywords}</div>}
                        {record.summary && <div><strong>{t('database.summary')}:</strong> {record.summary}</div>}
                    </div>
                );

                return (
                    <Tooltip title={tooltipContent} placement="topLeft">
                        <div>
                            {vertices.map((vertex, index) => (
                                <Tag key={index} color="green" style={{ margin: '2px' }}>
                                    {vertex}
                                </Tag>
                            ))}
                        </div>
                    </Tooltip>
                );
            },
        },
        {
            title: t('database.description'),
            dataIndex: 'description',
            key: 'description',
            width: 200,
            render: (text) => text || '-',
        },
        {
            title: t('database.action'),
            key: 'action',
            width: 100,
            render: (_, record) => (
                <Space size="small">
                    <Button
                        type="link"
                        icon={<EyeOutlined />}
                        onClick={() => openModal('view', 'hyperedge', record.hyperedge_id)}
                    >
                        {t('database.view')}
                    </Button>
                    <Button
                        type="link"
                        icon={<EditOutlined />}
                        onClick={() => openModal('edit', 'hyperedge', record.hyperedge_id)}
                    >
                        {t('database.edit')}
                    </Button>
                    <Popconfirm
                        title={t('database.confirm_delete_hyperedge')}
                        onConfirm={() => handleDeleteHyperedge(record.hyperedge_id)}
                        okText={t('database.confirm')}
                        cancelText={t('database.cancel')}
                    >
                        <Button type="link" danger icon={<DeleteOutlined />}>
                            {t('database.delete')}
                        </Button>
                    </Popconfirm>
                </Space>
            ),
        },
    ];

    // 处理vertices数据为表格格式
    const verticesTableData = vertices.map(vertex => ({
        key: vertex,
        vertex_id: vertex
    }));

    // 处理hyperedges数据为表格格式
    const hyperedgesTableData = hyperedges.map((hyperedge, index) => ({
        key: index,
        hyperedge_id: hyperedge.id || hyperedge,
        description: hyperedge.description || hyperedge.keywords || '-',
        keywords: hyperedge.keywords || '',
        summary: hyperedge.summary || ''
    }));

    // 数据库切换处理
    const onDatabaseChange = () => {
        // 重置分页
        setVerticesPagination({
            current: 1,
            pageSize: 10,
            total: 0
        });
        setHyperedgesPagination({
            current: 1,
            pageSize: 10,
            total: 0
        });
        fetchData(storeGlobalUser.selectedDatabase, true);
    };

    // 处理vertices分页变化
    const handleVerticesTableChange = (pagination) => {
        setVerticesPagination(pagination);
        // 重新请求数据
        fetchVerticesData(pagination.current, pagination.pageSize);
    };

    // 处理hyperedges分页变化
    const handleHyperedgesTableChange = (pagination) => {
        setHyperedgesPagination(pagination);
        // 重新请求数据
        fetchHyperedgesData(pagination.current, pagination.pageSize);
    };

    // 获取vertices数据
    const fetchVerticesData = async (page = 1, pageSize = 10) => {
        const database = storeGlobalUser.selectedDatabase;
        if (!database) {
return;
}

        setLoading(true);
        try {
            const verticesUrl = `${SERVER_URL}/db/vertices?database=${encodeURIComponent(database)}&page=${page}&page_size=${pageSize}`;
            const response = await fetch(verticesUrl);

            if (!response.ok) {
                throw new Error(`Vertices API failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (data.data) {
                setVertices(data.data);
                setVerticesPagination({
                    current: data.page,
                    pageSize: data.page_size,
                    total: data.total
                });
            } else {
                setVertices(data);
                setVerticesPagination({
                    current: 1,
                    pageSize: data.length,
                    total: data.length
                });
            }
        } catch (error) {
            console.error('Fetch vertices error:', error);
            message.error(t('database.fetch_data_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    // 获取hyperedges数据
    const fetchHyperedgesData = async (page = 1, pageSize = 10) => {
        const database = storeGlobalUser.selectedDatabase;
        if (!database) {
return;
}

        setLoading(true);
        try {
            const hyperedgesUrl = `${SERVER_URL}/db/hyperedges?database=${encodeURIComponent(database)}&page=${page}&page_size=${pageSize}`;
            const response = await fetch(hyperedgesUrl);

            if (!response.ok) {
                throw new Error(`Hyperedges API failed: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();

            if (data.data) {
                setHyperedges(data.data);
                setHyperedgesPagination({
                    current: data.page,
                    pageSize: data.page_size,
                    total: data.total
                });
            } else {
                setHyperedges(data);
                setHyperedgesPagination({
                    current: 1,
                    pageSize: data.length,
                    total: data.length
                });
            }
        } catch (error) {
            console.error('Fetch hyperedges error:', error);
            message.error(t('database.fetch_data_failed') + '：' + error.message);
        }
        setLoading(false);
    };

    return (
        <div>
            {/* 顶部数据库选择器 */}
            <Card style={{ marginBottom: 24 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                        <h3 style={{ margin: 0 }}>{t('database.title')}</h3>
                        <DatabaseSelector
                            mode="select"
                            // showRefresh={true}
                            size="middle"
                            onChange={onDatabaseChange}
                        />
                    </div>
                </div>

                {storeGlobalUser.selectedDatabase && (
                    <Row gutter={16} style={{ marginTop: 16 }}>
                        <Col span={6}>
                            <Statistic title={t('database.entity_count')} value={verticesPagination.total} prefix={<DatabaseOutlined />} />
                        </Col>
                        <Col span={6}>
                            <Statistic title={t('database.hyperedge_count')} value={hyperedgesPagination.total} prefix={<DatabaseOutlined />} />
                        </Col>
                    </Row>
                )}
            </Card>

            {/* 数据表格区域 */}
            {storeGlobalUser.selectedDatabase ? (
                <div className='flex  gap-4'>
                    {/* Vertices表格 */}
                    <Card title={t('database.vertices')} style={{ marginBottom: 24 }}
                        extra={
                            <Button
                                type="primary"
                                icon={<PlusOutlined />}
                                onClick={() => openModal('add', 'vertex')}
                            >
                                {t('database.add_vertex')}
                            </Button>
                        }
                    >

                        <Table
                            columns={vertexColumns}
                            dataSource={verticesTableData}
                            loading={loading}
                            pagination={verticesPagination}
                            onChange={handleVerticesTableChange}
                            scroll={{ x: 300 }}
                        />
                    </Card>

                    {/* Hyperedges表格 */}
                    <Card title={t('database.hyperedges')}
                        className='flex-1'
                        extra={
                            <Button
                                type="primary"
                                icon={<PlusOutlined />}
                                onClick={() => openModal('add', 'hyperedge')}
                            >
                                {t('database.add_hyperedge')}
                            </Button>
                        }
                    >
                        <Table
                            columns={hyperedgeColumns}
                            dataSource={hyperedgesTableData}
                            loading={loading}
                            pagination={hyperedgesPagination}
                            onChange={handleHyperedgesTableChange}
                            scroll={{ x: 400 }}
                        />
                    </Card>
                </div>
            ) : (
                <Card style={{ textAlign: 'center', padding: '60px 0' }}>
                    <DatabaseOutlined style={{ fontSize: '64px', color: '#d9d9d9', marginBottom: 16 }} />
                    <h3>{t('database.select_database_prompt')}</h3>
                    <p style={{ color: '#999' }}>{t('database.select_database_help')}</p>
                </Card>
            )}

            {/* Modal for Add/Edit/View */}
            <Modal
                title={
                    modalType === 'add'
                        ? t('database.add') + ' ' + (modalDataType === 'vertex' ? 'Vertex' : 'Hyperedge')
                        : modalType === 'edit'
                            ? t('database.edit') + ' ' + (modalDataType === 'vertex' ? 'Vertex' : 'Hyperedge')
                            : t('database.view') + ' ' + (modalDataType === 'vertex' ? 'Vertex' : 'Hyperedge')
                }
                open={modalVisible}
                onCancel={() => {
                    setModalVisible(false);
                    setModalLoading(false);
                    form.resetFields();
                }}
                footer={modalType === 'view' ? [
                    <Button key="close" onClick={() => {
                        setModalVisible(false);
                        setModalLoading(false);
                    }}>
                        {t('database.close')}
                    </Button>
                ] : [
                    <Button key="cancel" onClick={() => {
                        setModalVisible(false);
                        setModalLoading(false);
                    }}>
                        {t('database.cancel')}
                    </Button>,
                    <Button
                        key="submit"
                        type="primary"
                        loading={loading}
                        onClick={() => form.submit()}
                    >
                        {modalType === 'add' ? t('database.add') : t('database.update')}
                    </Button>
                ]}
                width={modalType === 'view' && modalDataType === 'vertex' ? 1200 : 600}
            >
                <Spin spinning={modalLoading} tip={t('database.loading_data')}>
                    {modalType === 'view' && modalDataType === 'vertex' ? (
                        // 查看Vertex时并列显示详细信息和超图
                        <div style={{ display: 'flex', gap: '20px', height: '500px' }}>
                            {/* 左侧：详细信息 */}
                            <div style={{ flex: '0 0 400px', overflowY: 'auto' }}>
                                <Card title={t('database.detail_info')} size="small" style={{ height: '100%' }}>
                                    <Form
                                        form={form}
                                        layout="vertical"
                                        size="small"
                                    >
                                        <Form.Item
                                            name="vertex_id"
                                            label={t('database.vertex_id')}
                                        >
                                            <Input disabled />
                                        </Form.Item>
                                        <Form.Item name="entity_name" label={t('database.entity_name')}>
                                            <Input disabled />
                                        </Form.Item>
                                        <Form.Item name="entity_type" label={t('database.entity_type')}>
                                            <Input disabled />
                                        </Form.Item>
                                        <Form.Item name="description" label={t('database.description')}>
                                            <TextArea rows={4} disabled />
                                        </Form.Item>
                                        <Form.Item name="additional_properties" label={t('database.additional_properties')}>
                                            <TextArea rows={4} disabled />
                                        </Form.Item>
                                    </Form>
                                </Card>
                            </div>

                            {/* 右侧：关系图谱 */}
                            <div style={{ flex: 1, minWidth: 0 }}>
                                <Card title={t('database.relation_graph')} size="small" style={{ height: '100%' }}>
                                    <div style={{ height: 'calc(100% - 40px)' }}>
                                        <HyperGraph
                                            vertexId={selectedRecord}
                                            height="100%"
                                            width="100%"
                                            showTooltip={true}
                                            graphId={`vertex-graph-${selectedRecord}`}
                                            database={storeGlobalUser.selectedDatabase}
                                        />
                                    </div>
                                </Card>
                            </div>
                        </div>
                    ) : (
                        // 其他情况（添加/编辑）显示正常表单
                        <Form
                            form={form}
                            layout="vertical"
                            onFinish={modalDataType === 'vertex' ? handleVertexSubmit : handleHyperedgeSubmit}
                        >
                            {modalDataType === 'vertex' ? (
                                <>
                                    <Form.Item
                                        name="vertex_id"
                                        label={t('database.vertex_id')}
                                        rules={[{ required: modalType === 'add', message: t('database.vertex_id_required') }]}
                                    >
                                        <Input
                                            placeholder={t('database.enter_vertex_id')}
                                            disabled={modalType !== 'add'}
                                        />
                                    </Form.Item>
                                    <Form.Item name="entity_name" label={t('database.entity_name')}>
                                        <Input placeholder={t('database.enter_entity_name')} disabled={modalType === 'view'} />
                                    </Form.Item>
                                    <Form.Item name="entity_type" label={t('database.entity_type')}>
                                        <Input placeholder={t('database.enter_entity_type')} disabled={modalType === 'view'} />
                                    </Form.Item>
                                    <Form.Item name="description" label={t('database.description')}>
                                        <TextArea
                                            rows={3}
                                            placeholder={t('database.enter_description')}
                                            disabled={modalType === 'view'}
                                        />
                                    </Form.Item>
                                    <Form.Item name="additional_properties" label={t('database.additional_properties')}>
                                        <TextArea
                                            rows={3}
                                            placeholder={t('database.enter_additional_properties')}
                                            disabled={modalType === 'view'}
                                        />
                                    </Form.Item>
                                </>
                            ) : (
                                <>
                                    {modalType === 'view' && (
                                        <Card
                                            title={t('database.hyperedge_info')}
                                            size="small"
                                            style={{ marginBottom: '16px' }}
                                        >
                                            <Descriptions size="small" column={1}>
                                                <Descriptions.Item label={t('database.contained_vertices')}>
                                                    {selectedRecord && selectedRecord.split('|*|').map((vertex, index) => (
                                                        <Tag key={index} color="blue" style={{ margin: '2px' }}>
                                                            {vertex}
                                                        </Tag>
                                                    ))}
                                                </Descriptions.Item>
                                            </Descriptions>
                                        </Card>
                                    )}

                                    <Form.Item
                                        name="vertices"
                                        label={t('database.vertices_title')}
                                        rules={[{ required: modalType === 'add', message: t('database.vertices_required') }]}
                                    >
                                        <Input
                                            placeholder={t('database.enter_vertices')}
                                            disabled={modalType === 'edit' || modalType === 'view'}
                                        />
                                    </Form.Item>
                                    <Form.Item name="keywords" label={t('database.keywords')}>
                                        <Input
                                            placeholder={t('database.enter_keywords')}
                                            disabled={modalType === 'view'}
                                        />
                                    </Form.Item>
                                    <Form.Item name="summary" label={t('database.summary')}>
                                        <TextArea
                                            rows={3}
                                            placeholder={t('database.enter_summary')}
                                            disabled={modalType === 'view'}
                                        />
                                    </Form.Item>
                                </>
                            )}
                        </Form>
                    )}
                </Spin>
            </Modal>
        </div>
    );
};

export default observer(HyperDB);
