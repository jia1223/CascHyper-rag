import React, { useState } from 'react'
import { Card, Alert, Spin } from 'antd'
import { SERVER_URL } from '../../utils'

const APIPage = () => {
    const [loading, setLoading] = useState(true)
    const [error, setError] = useState(false)

    const handleLoad = () => {
        setLoading(false)
    }

    const handleError = () => {
        setLoading(false)
        setError(true)
    }

    return (
        <div className='h-screen'
        >
            {error && (
                <Alert
                    message="无法加载API文档"
                    description={`请确保后端服务已启动并运行在 ${SERVER_URL}`}
                    type="error"
                    showIcon
                    style={{ margin: 16 }}
                />
            )}

            {loading && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    alignItems: 'center',
                    height: '100%'
                }}>
                    <Spin size="large" />
                </div>
            )}

            <iframe
                src={`${SERVER_URL}/docs`}
                style={{
                    width: '100%',
                    height: '100%',
                    border: 'none',
                    display: error ? 'none' : 'block'
                }}
                onLoad={handleLoad}
                onError={handleError}
                title="API Documentation"
            />
        </div>
    )
}

export default APIPage