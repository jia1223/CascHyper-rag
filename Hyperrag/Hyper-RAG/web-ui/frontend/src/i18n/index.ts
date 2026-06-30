import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'
import zhCN from './locales/zh-CN.json'
import enUS from './locales/en-US.json'

const resources = {
  'zh-CN': {
    translation: zhCN
  },
  'en-US': {
    translation: enUS
  }
}

i18n.use(initReactI18next).init({
  resources,
  lng: localStorage.getItem('language') || 'en-US', // 从本地存储获取语言设置，默认中文
  fallbackLng: 'en-US',
  interpolation: {
    escapeValue: false
  }
})

export default i18n
