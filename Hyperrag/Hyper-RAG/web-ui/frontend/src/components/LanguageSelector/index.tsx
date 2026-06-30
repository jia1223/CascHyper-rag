import React from 'react'
import { Select } from 'antd'
import { useTranslation } from 'react-i18next'
import { GlobalOutlined } from '@ant-design/icons'

const { Option } = Select

const LanguageSelector: React.FC = () => {
  const { i18n, t } = useTranslation()

  const handleLanguageChange = (value: string) => {
    i18n.changeLanguage(value)
    localStorage.setItem('language', value)
  }

  return (
    <Select
      value={i18n.language}
      onChange={handleLanguageChange}
      style={{ width: 150 }}
      suffixIcon={<GlobalOutlined />}
    >
      <Option value="zh-CN">{t('common.chinese')}</Option>
      <Option value="en-US">{t('common.english')}</Option>
    </Select>
  )
}

export default LanguageSelector
